// =============================================================================
// rms_norm_binary_pto.hpp — RMSNorm for g_r > tile_r (R-split)
// =============================================================================
//
// tiling[5] = {g_a, g_r, tile_a, tile_r, pow_r}
// pow_r: 2^k,  pow_r < g_r <= 2 * pow_r;  rem_r = g_r - pow_r
//
// Pass1 三段：
//   1) rem_r 按 tile_r 对齐的整块，区间 [0, n_rem_full*tile_r)，折叠 +pow_r
//        load lo/hi → cast → square → TADD(sq) → TROWSUM → acc(sum)
//   2) rem_r 对齐后的尾块（若 rem_tail>0），折叠 +pow_r
//        同上，ValidCol = rem_tail
//   3) pow_r 剩下部分 head_r = pow_r - rem_r，区间 [rem_r, pow_r)
//        load → cast → square → TROWSUM → acc(sum)
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_BINARY_PTO_HPP
#define SUPERNPU_RMS_NORM_BINARY_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_bin {

constexpr int kVecCols = 32;

// TEPL TADD with runtime Valid (upstream TADD still uses NTTP B.DIM immediates).
template <typename Tile>
inline void tadd(Tile &dst, Tile &src0, Tile &src1) {
    const size_t valid_col = src0.GetValidCol();
    const size_t valid_row = src0.GetValidRow();
    asm volatile(
        "BSTART.TEPL 0, %c1\n"
        "B.DIM %2, 0, ->lb0\n"
        "B.DIM %3, 0, ->lb1\n"
        "B.DIM zero, %c4, ->lb2\n"
        "B.IOT %5, %6, mask=15, TSize=%c7, last, ->%0\n"
        ""
        : "=Tr"(dst.data())
        : "i"(type_traits<typename Tile::DType>::TypeCode),
          "r"(valid_col),
          "r"(valid_row),
          "i"(Tile::Cols),
          "Tr"(src0.data()),
          "Tr"(src1.data()),
          "i"(tile_type_traits<typename Tile::TileDType>::TilesizeCode));
}

template <typename TileVec>
inline void rsqrt_newton(TileVec &out, TileVec &a) {
    TileVec x, t1, t2;
    TRECIP(x, a);
    for (int64_t i = 0; i < 4; ++i) {
        TMUL(t1, x, x);
        TMUL(t2, t1, a);
        TMULS(t2, t2, -0.5f);
        TADDS(t2, t2, 1.5f);
        TMUL(x, x, t2);
    }
    TMULS(out, x, 1.0f);
}

} // namespace rms_bin

template <typename dtype>
void rms_norm_binary(dtype *x, const int64_t *tiling, dtype *out,
                     float *workspace, float eps = 1e-6f) {
    (void)workspace;

    constexpr int64_t tA = 1;
    constexpr int64_t tR = 1024;

    const int64_t gA = tiling[0];
    const int64_t gR = tiling[1];
    const int64_t tile_a = tiling[2] > 0 ? tiling[2] : tA;
    const int64_t tile_r = tiling[3] > 0 ? tiling[3] : tR;
    const int64_t powR = tiling[4];
    (void)tile_a;

    const int64_t remR = gR - powR;
    const int64_t headR = powR - remR;
    // Pass1 remR：tile_r 整块数 + 尾块长
    const int64_t n_rem_full = remR / tile_r;
    const int64_t rem_tail = remR - n_rem_full * tile_r;
    // Pass2 整行
    const int64_t n_full = gR / tile_r;
    const int64_t tail_r = gR - n_full * tile_r;
    const float inv_r = 1.0f / static_cast<float>(gR);

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_v = Tile<Location::Vec, float, tA, rms_bin::kVecCols,
                        BLayout::RowMajor, 1, 1>;

    for (int64_t ia = 0; ia < gA; ++ia) {
        constexpr size_t active_a = 1;
        const size_t full_r = static_cast<size_t>(tile_r);

        tile_v cur, sum, mean, denom, rms;
        TEXPANDS(sum, 0.0f);

        // =====================================================================
        // Pass1-1：remR 按 tile_r 对齐的整块，折叠 [0,·) 与 [powR,·)
        // =====================================================================
        for (int64_t tr = 0; tr < n_rem_full; ++tr) {
            const int64_t offset = ia * gR + tr * tile_r;
            gm_t gi0(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t gi1(x + offset + powR, static_cast<int>(gA),
                     static_cast<int>(gR));
            tile_h src0_h(active_a, full_r);
            tile_h src1_h(active_a, full_r);
            tile_f src0(active_a, full_r);
            tile_f src1(active_a, full_r);
            tile_f sq0(active_a, full_r);
            tile_f sq1(active_a, full_r);

            TLOAD(src0_h, gi0);
            TLOAD(src1_h, gi1);
            TCVT(src0, src0_h);
            TCVT(src1, src1_h);
            TMUL(sq0, src0, src0);
            TMUL(sq1, src1, src1);
            rms_bin::tadd(sq0, sq0, sq1);
            TROWSUM(cur, sq0);
            TADD(sum, sum, cur);
        }

        // =====================================================================
        // Pass1-2：remR 对齐后的尾块（ValidCol = rem_tail）
        // =====================================================================
        if (rem_tail > 0) {
            const int64_t offset = ia * gR + n_rem_full * tile_r;
            const size_t ar = static_cast<size_t>(rem_tail);
            gm_t gi0(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t gi1(x + offset + powR, static_cast<int>(gA),
                     static_cast<int>(gR));
            tile_h src0_h(active_a, ar);
            tile_h src1_h(active_a, ar);
            tile_f src0(active_a, ar);
            tile_f src1(active_a, ar);
            tile_f sq0(active_a, ar);
            tile_f sq1(active_a, ar);

            TLOAD(src0_h, gi0);
            TLOAD(src1_h, gi1);
            TCVT(src0, src0_h);
            TCVT(src1, src1_h);
            TMUL(sq0, src0, src0);
            TMUL(sq1, src1, src1);
            rms_bin::tadd(sq0, sq0, sq1);
            TROWSUM(cur, sq0);
            TADD(sum, sum, cur);
        }

        // =====================================================================
        // Pass1-3：powR 剩下部分 headR = powR - remR，区间 [remR, powR)
        // =====================================================================
        {
            const int64_t r0 = remR;
            const int64_t n_head_full = headR / tile_r;
            const int64_t head_tail = headR - n_head_full * tile_r;
            for (int64_t tr = 0; tr < n_head_full; ++tr) {
                const int64_t offset = ia * gR + r0 + tr * tile_r;
                gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
                tile_h src_h(active_a, full_r);
                tile_f src(active_a, full_r);
                tile_f sq(active_a, full_r);
                TLOAD(src_h, gi);
                TCVT(src, src_h);
                TMUL(sq, src, src);
                TROWSUM(cur, sq);
                TADD(sum, sum, cur);
            }
            if (head_tail > 0) {
                const int64_t offset = ia * gR + r0 + n_head_full * tile_r;
                const size_t ar = static_cast<size_t>(head_tail);
                gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
                tile_h src_h(active_a, ar);
                tile_f src(active_a, ar);
                tile_f sq(active_a, ar);
                TLOAD(src_h, gi);
                TCVT(src, src_h);
                TMUL(sq, src, src);
                TROWSUM(cur, sq);
                TADD(sum, sum, cur);
            }
        }

        // Pass1.5
        TMULS(mean, sum, inv_r);
        TADDS(denom, mean, eps);
        rms_bin::rsqrt_newton(rms, denom);

        // Pass2：整行整块 + 尾块
        for (int64_t tr = 0; tr < n_full; ++tr) {
            const int64_t offset = ia * gR + tr * tile_r;
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t go(out + offset, static_cast<int>(gA), static_cast<int>(gR));
            tile_h src_h(active_a, full_r);
            tile_h dst_h(active_a, full_r);
            tile_f src(active_a, full_r);
            tile_f dst(active_a, full_r);
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TROWEXPANDMUL(dst, src, rms);
            TCVT(dst_h, dst);
            TSTORE(go, dst_h);
        }
        if (tail_r > 0) {
            const int64_t offset = ia * gR + n_full * tile_r;
            const size_t ar = static_cast<size_t>(tail_r);
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t go(out + offset, static_cast<int>(gA), static_cast<int>(gR));
            tile_h src_h(active_a, ar);
            tile_h dst_h(active_a, ar);
            tile_f src(active_a, ar);
            tile_f dst(active_a, ar);
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TROWEXPANDMUL(dst, src, rms);
            TCVT(dst_h, dst);
            TSTORE(go, dst_h);
        }
    }
}

#endif // SUPERNPU_RMS_NORM_BINARY_PTO_HPP
