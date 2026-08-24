#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

// Four-PE multi-thread conv2d 1x1 with RowMajor global tensors, modeled on
// matmul_multithread.hpp.
//
// Mathematical semantics (per PE, batch of 4 inputs):
//   C_pe = A_pe * B   (conv2d 1x1 = matrix multiply)
//   A_pe (input):  RowMajor<gM, gK>  (NHWC flattened, gM = in_h*in_w, gK = in_c)
//   B (weight):    RowMajor<gK, gN>  (W^T, shared across all PEs)
//   C_pe (output): RowMajor<gM, gN>  (NHWC flattened)
//
// Host-visible storage:
//   - input is an array of four PE matrices, each with shape [gM, gK].
//   - output is an array of four PE matrices, each with shape [gM, gN].
//   - weight is one shared matrix with shape [gK, gN].
//
// Thread/PE mapping (batch-level parallelism):
//   - get_thread_idx() selects one complete input/output matrix pair from
//     those arrays: 4 inputs convolved with the same weight.
//   - The kernel's gM and tM are already PE-local dimensions; no further
//     row splitting occurs inside the kernel.
//   - weight is not split. Each PE loads the complete [tK, tN] rhs operand
//     into TileRight and publishes it to compiler-managed shared storage.
//
// Tile mapping:
//   - Each PE holds A_pe [tM, tK] and C_pe [tM, tN].
//   - The four PE-local A cells collectively form A_big [4*tM, tK].
//   - Each PE presents one TileRight B operand with shape [tK, tN].
//   - TMATMUL collectively computes C_big [4*tM, tN], while each PE receives
//     only its own accumulator C_pe [tM, tN].
//
// Difference from conv2d_rm_shared:
//   - input/output are arrays of four PE matrices (batch parallelism) instead
//     of one matrix split by tid inside the kernel.
//   - A is a PE-private full [tM, tK] Left tile (not a SharedTile).
//   - B is loaded into a local TileRight, then published to shared storage
//     via TMOV_L2S_PUBLISH (matmul_multithread pattern) instead of the
//     direct GM->Shared TLOAD.
//   - C is a PE-private full [tM, tN] Vec tile (not a tM/4 row slice).
template <typename dtype,
          const int in_c, const int in_h, const int in_w,
          const int out_c,
          const int tM, const int tN, const int tK>
void conv2d_1x1_rm_mt(float *output_ptr, dtype *input_nchw_ptr, dtype *weight_ptr) {
    constexpr int kPeNum = 4;
    constexpr int kTileByteLimit = 8 * 1024;

    static_assert(in_c == 1 * 1 * in_c, "conv2d_1x1 expects kh=kw=1");

    constexpr int gM = in_h * in_w;
    constexpr int gN = out_c;
    constexpr int gK = in_c;

    static_assert(gM % tM == 0, "gM (in_h*in_w) must be divisible by tM");
    static_assert(gN % tN == 0, "gN (out_c) must be divisible by tN");
    static_assert(gK % tK == 0, "gK (in_c) must be divisible by tK");
    static_assert(tM * tK * sizeof(dtype) < kTileByteLimit,
                  "each PE A tile must be smaller than 8 KB");
    static_assert(tM * tN * sizeof(float) < kTileByteLimit,
                  "each PE C tile must be smaller than 8 KB");
    static_assert(tK * tN * sizeof(dtype) < kTileByteLimit,
                  "shared B tile must be smaller than 8 KB");

    const uint32_t tid = get_thread_idx();

    // input/output are arrays of PE matrices. Select one complete matrix
    // before constructing the PE-local global iterators. weight keeps its
    // shared base.
    input_nchw_ptr += tid * gM * gK;
    output_ptr += tid * gM * gN;

    using gm_shapeInput  = global_tensor<dtype, RowMajor<gM, gK>>;
    using gm_shapeWeight = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeOutput = global_tensor<float, RowMajor<gM, gN>>;

    // PE-private lhs and output cells.
    using tileA = TileLeft<dtype, tM, tK>;
    using tileC =
        Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;

    // TLOAD requires a local tile. Publish it to compiler-managed shared
    // storage before passing B to the shared-right TMATMUL overload.
    using tileBLocal = TileRight<dtype, tK, tN>;
    using tileBShared = SharedTile<tileBLocal>;

    using itA = global_iterator<gm_shapeInput, tileA>;
    using itB = global_iterator<gm_shapeWeight, tileBLocal>;
    using itC = global_iterator<gm_shapeOutput, tileC>;

    itA gAIter(input_nchw_ptr);
    itB gBIter(weight_ptr);
    itC gCIter(output_ptr);

    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    #pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; ++j) {
            tileC tC;

            if constexpr (Kb == 1) {
                tileA tA;
                tileBLocal tBLocal;

                auto gA = gAIter(i, 0);
                TLOAD(tA, gA);
                auto gB = gBIter(0, j);
                TLOAD(tBLocal, gB);
                tileBShared tBShared = TMOV_L2S_PUBLISH(tBLocal);
                TMATMUL(tC, tA, tBShared);
            } else {
                {
                    tileA tA;
                    tileBLocal tBLocal;
                    auto gA = gAIter(i, 0);
                    auto gB = gBIter(0, j);
                    TLOAD(tA, gA);
                    TLOAD(tBLocal, gB);
                    tileBShared tBShared = TMOV_L2S_PUBLISH(tBLocal);
                    TMATMUL(tC, tA, tBShared);
                }

                #pragma clang loop unroll(full)
                for (int k = 1; k < Kb; ++k) {
                    tileA tA;
                    tileBLocal tBLocal;
                    auto gA = gAIter(i, k);
                    auto gB = gBIter(k, j);
                    TLOAD(tA, gA);
                    TLOAD(tBLocal, gB);
                    tileBShared tBShared = TMOV_L2S_PUBLISH(tBLocal);
                    TMATMUL_ACC(tC, tC, tA, tBShared);
                }
            }

            auto gC = gCIter(i, j);
            TSTORE(gC, tC);
        }
    }
}
