// =============================================================================
// rms_norm_pto.hpp — RMSNorm (one-level PTO)
// =============================================================================
//
//   out[row] = x[row] * rsqrt(mean(x[row]^2) + eps)
//
// Entry:
//   rms_norm<dtype>(x, tiling, out, eps);
//   tiling[4] = {g_m, g_n, tile_m, tile_n}  (int64_t)
//
// Pipeline (fp16 in/out, fp32 compute):
//   TLOAD → TCVT → TMUL(x,x) → TROWSUM → TMULS(1/N) → TADDS(eps)
//   → Newton rsqrt → TROWEXPANDMUL → TCVT → TSTORE
//
// Tile ValidRow/Col are compile-time (one-level ISA immediates). g_m / g_n
// come from tiling at runtime. Current Tile shape: (1, 512).
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_PTO_HPP
#define SUPERNPU_RMS_NORM_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace detail {

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

} // namespace detail

// tiling: [g_m, g_n, tile_m, tile_n]
template <typename dtype>
void rms_norm(dtype *x, const int64_t *tiling, dtype *out, float eps = 1e-6f) {
    constexpr int64_t tM = 1;
    constexpr int64_t tN = 1024;

    const int64_t gM = tiling[0];
    const int64_t gN = tiling[1];

    // const int64_t tile_m = tiling[2];
    // const int64_t tile_n = gN;
    // using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    // using tile_h = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, -1, -1>;
    // using tile_f = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor, -1, -1>;
    // using tile_v = Tile<Location::Vec, float, tM, 8, BLayout::RowMajor, -1, 1>;
    /*以下写法能好*/
    const int64_t tile_m = 1;
    const int64_t tile_n = 512;
    using gm_t = global_tensor<dtype, RowMajor<16, 512>>;
    using tile_h = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, 1, 512>;
    using tile_f = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor, 1, 512>;
    using tile_v = Tile<Location::Vec, float, tM, 8, BLayout::RowMajor, 1, 1>;

    const float inv_n = 1.0f / static_cast<float>(gN);

    

    for (int64_t i = 0; i < gM; i += tile_m) {
        const int64_t offset = i * gN;
        // RowMajor<16,512> is static; only pointer ctor is valid.
        gm_t gi(x + offset);
        gm_t go(out + offset);

        tile_h src_h, dst_h;
        tile_f src, squared, dst;
        tile_v sqrsum, mean, denom, rms;

        // tile_h src_h(0, 1, 512), dst_h(0, 1, 512);
        // tile_f src(0, 1, 512), squared(0, 1, 512), dst(0, 1, 512);
        // tile_v sqrsum(0, 1), mean(0, 1), denom(0, 1), rms(0, 1);

        // tile_h src_h(0, tile_m, tile_n), dst_h(0, tile_m, tile_n);
        // tile_f src(0, tile_m, tile_n), squared(0, tile_m, tile_n), dst(0, tile_m, tile_n);
        // tile_v sqrsum(0, tile_m), mean(0, tile_m), denom(0, tile_m), rms(0, tile_m);

        TLOAD(src_h, gi);
        TCVT(src, src_h);
        TMUL(squared, src, src);
        TROWSUM(sqrsum, squared);
        TMULS(mean, sqrsum, inv_n);
        TADDS(denom, mean, eps);
        detail::rsqrt_newton(rms, denom);
        TROWEXPANDMUL(dst, src, rms);
        TCVT(dst_h, dst);
        TSTORE(go, dst_h);
    }
}

#endif // SUPERNPU_RMS_NORM_PTO_HPP
