#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

// Four-PE low-precision matmul.
//
// A/B are Core-shared operands. Each PE owns one [tM / 4, tN] FP32 C tile.
// PackedFactor is 1 for 8-bit formats and 2 for packed FP4x2 formats.
// Scaled modes (HiF8/HiF4x2/MXFP8/MXFP4) additionally load
// per-32-logical-element E8M0 scale data. For packed FP4x2, this is one byte
// per 16 stored elements:
//   AScale: [M, K / 32], partitioned by C's PE-local M rows.
//   BScale: [K / 32, N], replicated in each PE for the shared B block.
template <typename dtype, int PackedFactor, bool UseMx,
          int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_shared_lowp(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                        uint8_t *a_scale_ptr, uint8_t *b_scale_ptr) {
    constexpr int kPeNum = 4;
    constexpr int kPeM = tM / kPeNum;
    constexpr int kStoredGK = gK / PackedFactor;
    constexpr int kStoredTK = tK / PackedFactor;
    constexpr int kScaleGroup = 32;

    static_assert(PackedFactor == 1 || PackedFactor == 2,
                  "PackedFactor must be 1 (FP8) or 2 (FP4x2)");
    static_assert(gM % tM == 0, "M must be divisible by tM");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(gK % PackedFactor == 0 && tK % PackedFactor == 0,
                  "K must be divisible by the packed element factor");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(!UseMx || (gK % kScaleGroup == 0 &&
                             tK % kScaleGroup == 0),
                  "MX formats require K and tK divisible by 32");

    const uint32_t tid = get_thread_idx();

    using gmA = global_tensor<dtype, RowMajor<gM, kStoredGK>>;
    using gmB = global_tensor<dtype, RowMajor<kStoredGK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileALocal = TileLeft<dtype, tM, kStoredTK>;
    using tileBLocal = TileRight<dtype, kStoredTK, tN>;
    using tileAShared = SharedTile<tileALocal>;
    using tileBShared = SharedTile<tileBLocal>;
    using tileC =
        Tile<Location::Vec, float, kPeM, tN, BLayout::RowMajor>;

    using itA = global_iterator<gmA, tileALocal>;
    using itB = global_iterator<gmB, tileBLocal>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    // Scaling tiles use the matrix operand's physical K extent while their
    // valid shape contains one E8M0 scale byte per 32 logical K elements.
    using gmAScale =
        global_tensor<uint8_t, RowMajor<gM, gK / kScaleGroup>>;
    using gmBScale =
        global_tensor<uint8_t, RowMajor<gK / kScaleGroup, gN>>;
    using tileAScale =
        Tile<Location::Scaling, uint8_t, kPeM, kStoredTK,
             BLayout::RowMajor, kPeM, tK / kScaleGroup,
             SLayout::RowMajor>;
    using tileBScale =
        Tile<Location::Scaling, uint8_t, kStoredTK, tN,
             BLayout::ColMajor, tK / kScaleGroup, tN,
             SLayout::ColMajor>;
    using itAScale = global_iterator<gmAScale, tileAScale>;
    using itBScale = global_iterator<gmBScale, tileBScale>;

    itAScale gIterAScale(a_scale_ptr);
    itBScale gIterBScale(b_scale_ptr);

    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;

#pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
#pragma clang loop unroll(full)
        for (int j = 0; j < Nb; ++j) {
            tileC tC;

#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                tileAShared tA;
                tileBShared tB;
                auto gA = gIterA(i, k);
                auto gB = gIterB(k, j);
                TLOAD<tileALocal, 1>(tA, gA);
                TLOAD<tileBLocal, 1>(tB, gB);

                if constexpr (UseMx) {
                    tileAScale tAScale;
                    tileBScale tBScale;
                    auto gAScale = gIterAScale(i * kPeNum + tid, k);
                    auto gBScale = gIterBScale(k, j);
                    TLOAD(tAScale, gAScale);
                    TLOAD(tBScale, gBScale);

                    if (k == 0) {
                        TMATMUL_MX(tC, tA, tAScale, tB, tBScale);
                    } else {
                        TMATMUL_MX_ACC(tC, tC, tA, tAScale, tB, tBScale);
                    }
                } else {
                    if (k == 0) {
                        TMATMUL(tC, tA, tB);
                    } else {
                        TMATMUL_ACC(tC, tC, tA, tB);
                    }
                }
            }

            auto gC = gIterC(i * kPeNum + tid, j);
            TSTORE(gC, tC);
        }
    }
}
