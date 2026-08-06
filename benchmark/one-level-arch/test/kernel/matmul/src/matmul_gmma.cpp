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
//   Big logical tile across four PEs:
//     A_big: [4*tM, tK]
//     B_shared: [tK, tN]
//     C_big: [4*tM, tN]
//
//   PE-local tile:
//     A_pe: [tM, tK]
//     C_pe: [tM, tN]
//
//   Right operand:
//     B is not split by PE. For every (K, N) block, one shared B tile
//     [tK, tN] is loaded and consumed by the collective gmma.
//
// Execution model:
//   - get_thread_idx() selects one complete PE-local A/C matrix.
//   - Each PE independently loads its A tile and the same logical B tile.
//   - The four PE-local products collectively form the big logical result.
//   - Each PE stores only its own C matrix.
//
// This file is mainly used to describe the GMMA tile mapping. The name gmma is
// intentionally kept as the intrinsic name from the programming model.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_mask_gmma_tileop(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kTileByteLimit = 4 * 1024;

    static_assert(gM % tM == 0, "gM must be divisible by tM in this GMMA model");
    static_assert(gN % tN == 0, "gN must be divisible by tN in this GMMA model");
    static_assert(gK % tK == 0, "gK must be divisible by tK in this GMMA model");
    static_assert(tM * tK * sizeof(dtype) < kTileByteLimit,
                  "each PE A tile must be smaller than 4 KB");
    static_assert(tM * tN * sizeof(float) < kTileByteLimit,
                  "each PE C tile must be smaller than 4 KB");
    static_assert(tK * tN * sizeof(dtype) < kTileByteLimit,
                  "B tile must be smaller than 4 KB");

    // gM/tM are PE-local dimensions. A and C are arrays of four complete
    // PE-local matrices; each hardware thread selects one matrix.
    const uint32_t tid = get_thread_idx();
    a_ptr += tid * gM * gK;
    c_ptr += tid * gM * gN;

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileA = TileLeft<dtype, tM, tK>;
    using tileB = TileRight<dtype, tK, tN>;
    using tileC = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;

    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileB>;
    using itC = global_iterator<gmC, tileC>;

    itA gAIter(a_ptr);
    itB gBIter(b_ptr);
    itC gCIter(c_ptr);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            tileC tC;

#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                tileA tA;
                tileB tB;
                tileC tPart;

                auto gA = gAIter(i, k);
                auto gB = gBIter(k, j);
                TLOAD(tA, gA);
                TLOAD(tB, gB);
                TMATMUL_FIXP(tPart, tA, tB, fixp::keep_acc());

                if (k == 0) {
                    tC = tPart;
                } else {
                    TADD(tC, tC, tPart);
                }
            }

            auto gC = gCIter(i, j);
            TSTORE(gC, tC);
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
    static_assert(globM % 4 == 0, "global M must be divisible by PE count");
    matmul_mask_gmma_tileop<dtype, globM / 4, globN, globK,
                            tilM, tilN, tilK>(
        dst, src0, src1);
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t *)dst, globM * globN * sizeof(float));
#endif

    return 0;
}
