#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

// Four-PE cooperative matmul that keeps Shared B tiles alive across M blocks.
//
// Loop order and reuse:
//   for each N block
//     for each M block
//       for each K block
//         load A for the current M block
//         load B only for the first M block, then reuse it
//         accumulate C = A * B
//
// A and B are cooperative SharedTile operands. Each PE owns one contiguous
// [tM / 4, tN] row slice of the FP32 output tile.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_shared_reuseB(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kPeNum = 4;
    constexpr int kMaxReuseBTiles = 24;

    static_assert(gM % tM == 0, "M must be divisible by tM");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");

    const uint32_t tid = get_thread_idx();

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileALocal = TileLeft<dtype, tM, tK>;
    using tileBLocal = TileRight<dtype, tK, tN>;
    using tileAShared = SharedTile<tileALocal>;
    using tileBShared = SharedTile<tileBLocal>;
    using tileC =
        Tile<Location::Vec, float, tM / kPeNum, tN, BLayout::RowMajor>;

    using itA = global_iterator<gmA, tileALocal>;
    using itB = global_iterator<gmB, tileBLocal>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    constexpr int kReuseK =
        Kb < kMaxReuseBTiles ? Kb : kMaxReuseBTiles;

#pragma clang loop unroll(full)
    for (int j = 0; j < Nb; ++j) {
        // These Shared B handles span the complete M loop. The first M block
        // fills them while later M blocks consume the same Shared versions.
        tileBShared tBReuse[kReuseK];

#pragma clang loop unroll(full)
        for (int i = 0; i < Mb; ++i) {
            tileC tC;

#pragma clang loop unroll(full)
            for (int k = 0; k < kReuseK; ++k) {
                tileAShared tAShared;
                auto gA = gIterA(i, k);
                TLOAD<tileALocal, 1>(tAShared, gA);

                if (i == 0) {
                    auto gB = gIterB(k, j);
                    TLOAD<tileBLocal, 1>(tBReuse[k], gB);
                }

                if (k == 0) {
                    TMATMUL(tC, tAShared, tBReuse[k]);
                } else {
                    TMATMUL_ACC(tC, tC, tAShared, tBReuse[k]);
                }
            }

            // K blocks beyond the reuse window preserve the original compute
            // path and reload both operands for every M block.
            if constexpr (kReuseK < Kb) {
#pragma clang loop unroll(full)
                for (int k = kReuseK; k < Kb; ++k) {
                    tileAShared tAShared;
                    tileBShared tBShared;
                    auto gA = gIterA(i, k);
                    auto gB = gIterB(k, j);
                    TLOAD<tileALocal, 1>(tAShared, gA);
                    TLOAD<tileBLocal, 1>(tBShared, gB);
                    TMATMUL_ACC(tC, tC, tAShared, tBShared);
                }
            }

            auto gC = gIterC(i * kPeNum + tid, j);
            TSTORE(gC, tC);
        }
    }
}
