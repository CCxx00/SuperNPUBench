#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"
#include "fileop.h"

using namespace pto;

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
#define tilM 64
#endif

#ifndef tilN
#define tilN 64
#endif

#ifndef tilK
#define tilK 64
#endif

#ifndef Batch
#define Batch 1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

// Four-PE multi-thread GMMA matmul.
//
// Mathematical semantics:
//   C = A * B
//   A: [M, K], B: [K, N], C: [M, N]
//
// Host-visible storage:
//   - A is an array of four PE matrices, each with shape [gM, gK].
//   - C is an array of four PE matrices, each with shape [gM, gN].
//   - B is one shared matrix with shape [gK, gN].
//
// Thread/PE mapping:
//   - get_thread_id() selects one complete A/C matrix from those arrays.
//   - The kernel's gM and tM are already PE-local dimensions; no further
//     row splitting occurs inside the kernel.
//   - B is not split. TLOAD places a complete [tK, tN] rhs tile in shared
//     staging storage for the collective TMATMUL.
//
// Tile mapping:
//   - Each PE holds A_pe [tM, tK] and C_pe [tM, tN].
//   - The four PE-local A cells collectively form A_big [4*tM, tK].
//   - One shared B tile has shape [tK, tN].
//   - TMATMUL collectively computes C_big [4*tM, tN], while each PE receives
//     only its own accumulator C_pe [tM, tN].
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_multithread(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kTileElemLimit = 8 * 1024;

    static_assert(gM % tM == 0, "M must be divisible by tM");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(tM * tK < kTileElemLimit,
                  "each PE A tile must be smaller than 8K elements");
    static_assert(tM * tN < kTileElemLimit,
                  "each PE C tile must be smaller than 8K elements");
    static_assert(tK * tN < kTileElemLimit,
                  "shared B tile must be smaller than 8K elements");

    const uint32_t tid = get_thread_id();

    // A/C are arrays of PE matrices. Select one complete matrix before
    // constructing the PE-local global iterators. B keeps its shared base.
    a_ptr += tid * gM * gK;
    c_ptr += tid * gM * gN;

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    // PE-private lhs and output cells.
    using tileA = TileLeft<dtype, tM, tK>;
    using tileCAcc = TileAcc<float, tM, tN>;
    using tileC =
        Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;

    // Full rhs tile in shared GMMA staging storage.
    using tileB = TileRight<dtype, tK, tN>;

    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileB>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            tileCAcc tCAcc;

            for (int k = 0; k < Kb; ++k) {
                tileA tA;
                tileB tB;

                // Every PE loads from its selected A matrix. The kernel only
                // sees the PE-local tile shape [tM, tK].
                auto gA = gIterA(i, k);
                TLOAD(tA, gA);

                // B is common to all PEs. The group load is represented once
                // per SPMD program and maps to shared GMMA rhs staging.
                auto gB = gIterB(k, j);
                TLOAD(tB, gB);

                if (k == 0) {
                    TMATMUL(tCAcc, tA, tB);
                } else {
                    TMATMUL_ACC(tCAcc, tA, tB);
                }
            }

            // Convert and store only the current PE's C row slice.
            tileC tC;
            ACCCVT(tC, tCAcc);
            auto gC = gIterC(i, j);
            TSTORE(gC, tC);
        }
    }
}

int main() {
    using dtype = float;
    constexpr int kPeNum = 4;

    static_assert(globM % kPeNum == 0,
                  "global M must be divisible by the PE count");

    dtype src0p[Batch * globM * globK + 2 * ALIGN];
    dtype src1p[Batch * globK * globN + 2 * ALIGN];
    float dstp[Batch * globM * globN + 2 * ALIGN];

    dtype *src0 =
        (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype *src1 =
        (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    float *dst =
        (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    readBinaryFile(SRC0_PATH, (uint8_t *)src0,
                   Batch * globM * globK * sizeof(dtype));
    readBinaryFile(SRC1_PATH, (uint8_t *)src1,
                   Batch * globK * globN * sizeof(dtype));
#endif

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        // src0/dst contain four consecutive PE matrices. The kernel receives
        // the PE-local M dimension and uses tid to select one complete matrix.
        matmul_multithread<dtype, globM / kPeNum, globN, globK, tilM, tilN,
                           tilK>(
            dst + b * globM * globN,
            src0 + b * globM * globK,
            src1 + b * globK * globN);
    }
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t *)dst,
                    Batch * globM * globN * sizeof(float));
#endif

    return 0;
}
