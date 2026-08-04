#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"
#include "fileop.h"

#ifndef globM
#define globM 256
#endif

#ifndef globN
#define globN 256
#endif

#ifndef globK
#define globK 256
#endif

#ifndef tilM
#define tilM 128
#endif

#ifndef tilN
#define tilN 128
#endif

#ifndef tilK
#define tilK 64
#endif

#ifndef Batch
#define Batch 1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

using namespace pto;

// 4-PE GMMA matmul programming model.
//
// Mathematical semantics:
//   C = A * B
//   A: [gM, gK], B: [gK, gN], C: [gM, gN]
//
// Tile organization:
//   Big logical tile:
//     A_big: [tM, tK]
//     B_shared: [tK, tN]
//     C_big: [tM, tN]
//
//   PE-local tile:
//     A_pe: [tM / PE, tK]
//     C_pe: [tM / PE, tN]
//
//   Right operand:
//     B is not split by PE. For every (K, N) block, one shared B tile
//     [tK, tN] is loaded and consumed by the collective gmma.
//
// Execution model:
//   - Each PE independently loads its A row slice with TLOAD.
//   - B is loaded once as a shared/full tile.
//   - One scalar gmma call computes the logical tile
//       A_big[tM,tK] * B_shared[tK,tN] -> C_big[tM,tN]
//     and writes PE-local accumulator slices C_pe.
//   - Each PE stores its own C row slice.
//
// This file is mainly used to describe the GMMA tile mapping. The name gmma is
// intentionally kept as the intrinsic name from the programming model.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_mask_gmma_tileop(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr uint32_t kPeNum = 4;
    constexpr int kPeRows = tM / kPeNum;
    constexpr int kPeTileElemLimit = 8 * 1024;
    constexpr int kUnionTileElemLimit = kPeNum * kPeTileElemLimit;

    // This GMMA mapping assumes full logical tiles. Remainder/tail tiles can be
    // added with the same split rule, but are intentionally omitted here to keep
    // the programming model clear.
    static_assert(gM % tM == 0, "gM must be divisible by tM in this GMMA model");
    static_assert(gN % tN == 0, "gN must be divisible by tN in this GMMA model");
    static_assert(gK % tK == 0, "gK must be divisible by tK in this GMMA model");
    static_assert(tM % kPeNum == 0, "tM must be divisible by PE count");
    static_assert(kPeRows * tK <= kPeTileElemLimit,
                  "each PE A tile must fit into 8K elements");
    static_assert(kPeRows * tN <= kPeTileElemLimit,
                  "each PE C tile must fit into 8K elements");
    static_assert(tK * tN <= kUnionTileElemLimit,
                  "shared B tile must fit into 32K union elements");

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    // A is split along M across PEs:
    //   tA[pe] valid shape [kPeRows, tK]
    using tileA = TileLeft<dtype, kPeRows, tK>;

    // B is shared, not split:
    //   tB valid shape [tK, tN]
    using tileB = TileRight<dtype, tK, tN>;

    // FIXP writes every PE-local product directly to an ordinary Tile.
    using tileC = Tile<Location::Vec, float, kPeRows, tN, BLayout::RowMajor>;

    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileB>;
    using itC = global_iterator<gmC, tileC>;

    itA gAIter(a_ptr);
    itB gBIter(b_ptr);
    itC gCIter(c_ptr);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;

    for (int b = 0; b < Batch; ++b) {
        (void)b;
        for (int i = 0; i < Mb; ++i) {
            for (int j = 0; j < Nb; ++j) {
                tileC tC[kPeNum];

                for (int k = 0; k < Kb; ++k) {
                    tileA tA[kPeNum];
                    tileB tB;

                    // Left matrix A:
                    //   PE pe loads:
                    //     A[i*tM + pe*kPeRows : i*tM + (pe+1)*kPeRows,
                    //       k*tK : (k+1)*tK]
                    //   tA[0..3] logically forms A_big [tM,tK].
#pragma clang loop unroll(full)
                    for (int pe = 0; pe < kPeNum; ++pe) {
                        auto gA = gAIter(i * kPeNum + pe, k);
                        TLOAD(tA[pe], gA);
                    }

                    // Right matrix B:
                    //   B is loaded once as one shared/full tile:
                    //     B[k*tK : (k+1)*tK, j*tN : (j+1)*tN]
                    //   It is consumed by gmma as the shared rhs tile.
                    auto gB = gBIter(k, j);
                    TLOAD(tB, gB);

                    // The v5 ISA no longer supports exporting an implicit ACC.
                    // Produce one ordinary FIXP Tile per K block and accumulate
                    // those FP32 partials explicitly with TADD.
#pragma clang loop unroll(full)
                    for (int pe = 0; pe < kPeNum; ++pe) {
                        tileC tPart;
                        TMATMUL_FIXP(
                            tPart, tA[pe], tB, fixp::keep_acc());
                        if (k == 0) {
                            tC[pe] = tPart;
                        } else {
                            TADD(tC[pe], tC[pe], tPart);
                        }
                    }
                }

                // Store each PE-local C slice:
                //   C[i*tM + pe*kPeRows : i*tM + (pe+1)*kPeRows,
                //     j*tN : (j+1)*tN]
#pragma clang loop unroll(full)
                for (int pe = 0; pe < kPeNum; ++pe) {
                    auto gC = gCIter(i * kPeNum + pe, j);
                    TSTORE(gC, tC[pe]);
                }
            }
        }
    }
}

int main() {
#if defined(MASK_FP32)
    using dtype = float;
#else
    using dtype = __half;
#endif

    dtype src0p[globM * globK + 2 * ALIGN];
    dtype src1p[globK * globN + 2 * ALIGN];
    float dstp[globM * globN + 2 * ALIGN];

    dtype *src0 = (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype *src1 = (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    float *dst = (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    readBinaryFile(SRC0_PATH, (uint8_t *)src0, globM * globK * sizeof(dtype));
    readBinaryFile(SRC1_PATH, (uint8_t *)src1, globK * globN * sizeof(dtype));
#endif

    BENCHSTART;
    matmul_mask_gmma_tileop<dtype, globM, globN, globK, tilM, tilN, tilK>(
        dst, src0, src1);
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t *)dst, globM * globN * sizeof(float));
#endif

    return 0;
}
