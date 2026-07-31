// =============================================================================
// rms_norm_pto.hpp — RMSNorm (one-level PTO)
// =============================================================================
//
//   out[row] = x[row] * rsqrt(mean(x[row]^2) + eps)
//
// Entry:
//   rms_norm<dtype>(x, tiling, out, eps);
//   tiling[4] = {g_m, g_n, tile_m, tile_n}  (int64_t)
//   tile_n <= 0 means use g_n (full-row tile).
//
// Pipeline (fp16 in/out, fp32 compute):
//   TLOAD → TCVT → TMUL(x,x) → TROWSUM → TMULS(1/N) → TADDS(eps)
//   → Newton rsqrt → TROWEXPANDMUL → TCVT → TSTORE
//
// Dynamic ValidRow/ValidCol: Tile Valid = -1, ctor passes runtime values.
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_PTO_HPP
#define SUPERNPU_RMS_NORM_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_detail {

inline int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

template <typename TileVec>
inline void rsqrt_newton(TileVec &out, TileVec &a) {
    // Dynamic Valid: TEPL reads GetValid* from src0. Temps must carry ValidRow.
    const size_t vr = static_cast<size_t>(a.GetValidRow());
    TileVec x(vr), t1(vr), t2(vr);
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

} // namespace rms_detail

// tiling: [g_m, g_n, tile_m, tile_n]
template <typename dtype>
void rms_norm(dtype *x, const int64_t *tiling, dtype *out, float eps = 1e-6f) {
    constexpr int64_t tM = 1;
    constexpr int64_t tN = 512;

    const int64_t gM = tiling[0];
    const int64_t gN = tiling[1];
    const int64_t tile_m = tiling[2] > 0 ? tiling[2] : tM;
    const int64_t tile_n = tiling[3] > 0 ? tiling[3] : gN;

    if (gM <= 0 || gN <= 0 || tile_m <= 0 || tile_n <= 0 ||
        tile_m > tM || tile_n > tN) {
        return;
    }

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor, -1, -1>;
    // ValidCol=1; Cols=32 → 128B PE-local (TSize min).
    using tile_v = Tile<Location::Vec, float, tM, 32, BLayout::RowMajor, -1, 1>;

    const float inv_n = 1.0f / static_cast<float>(gN);

    for (int64_t i = 0; i < gM; i += tile_m) {
        const int64_t active_row = rms_detail::min64(tile_m, gM - i);
        const int64_t active_col = tile_n;
        const int64_t offset = i * gN;

        gm_t gi(x + offset, static_cast<int>(gM), static_cast<int>(gN));
        gm_t go(out + offset, static_cast<int>(gM), static_cast<int>(gN));

        tile_h src_h(static_cast<size_t>(active_row),
                     static_cast<size_t>(active_col));
        tile_h dst_h(static_cast<size_t>(active_row),
                     static_cast<size_t>(active_col));
        tile_f src(static_cast<size_t>(active_row),
                   static_cast<size_t>(active_col));
        tile_f squared(static_cast<size_t>(active_row),
                       static_cast<size_t>(active_col));
        tile_f dst(static_cast<size_t>(active_row),
                   static_cast<size_t>(active_col));
        tile_v sqrsum(static_cast<size_t>(active_row));
        tile_v mean(static_cast<size_t>(active_row));
        tile_v denom(static_cast<size_t>(active_row));
        tile_v rms(static_cast<size_t>(active_row));

        TLOAD(src_h, gi);
        TCVT(src, src_h);
        TMUL(squared, src, src);
        TROWSUM(sqrsum, squared);
        TMULS(mean, sqrsum, inv_n);
        TADDS(denom, mean, eps);
        rms_detail::rsqrt_newton(rms, denom);
        TROWEXPANDMUL(dst, src, rms);
        TCVT(dst_h, dst);
        TSTORE(go, dst_h);
    }
}

#endif // SUPERNPU_RMS_NORM_PTO_HPP
