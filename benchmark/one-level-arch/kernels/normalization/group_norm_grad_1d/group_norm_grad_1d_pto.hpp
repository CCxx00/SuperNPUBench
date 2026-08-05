// =============================================================================
// group_norm_grad_1d_pto.hpp — GroupNorm backward, HxW == 1 (one-level PTO)
// =============================================================================
//
// Matches PyTorch aten GroupNorm1d Backward (group_norm_kernel.cu):
//   Stage A1  reduce → c2/c3  (per n,g)
//   Stage A2  dX = (rstd*gamma)*dY + c2*X + c3
//   Stage B   dgamma / dbeta
//
// Layout: X/dY/dX [N,C]; mean/rstd [N,G] fp32; gamma/dgamma/dbeta [C]
// All pointers required (dy,x,mean,rstd,gamma,dx,dgamma,dbeta).
//
// tiling[4] = {N, C, G, tile_d}
//   tile_d <= 0 → min(D, tD); channel R-split on dgamma/dbeta when D > tile_d.
//   Stage A requires D <= tD (one tile).
//
// Tile capacity tD=128 (min TSize). Reduce and dX are separate passes so large
// tiles do not stay live across both.
// =============================================================================
#ifndef SUPERNPU_GROUP_NORM_GRAD_1D_PTO_HPP
#define SUPERNPU_GROUP_NORM_GRAD_1D_PTO_HPP

#include "normalization/group_norm_grad/group_norm_grad_dyn_ops.hpp"

#include <cstdint>

namespace gn_grad_1d {

using gn_grad_dyn::tadd;
using gn_grad_dyn::texpands;
using gn_grad_dyn::trowexpandadd;
using gn_grad_dyn::tsub;

// ---------------------------------------------------------------------------
// Stage A1: channel reduce → c2/c3 for one (n, g)
//   scratch[2] = {c2, c3}
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void fused_params_group(dtype *dy, dtype *x, float *mean, float *rstd,
                               dtype *gamma, float *scratch, int64_t N,
                               int64_t C, int64_t G, int64_t D, int64_t n,
                               int64_t g, float s) {
    const int64_t ng = n * G + g;
    const int64_t c0 = g * D;
    const int64_t offset = n * C + c0;
    const size_t active_d = static_cast<size_t>(D);

    gm_h gdy(dy + offset, static_cast<int>(N), static_cast<int>(C));
    gm_h gx(x + offset, static_cast<int>(N), static_cast<int>(C));
    gm_f gmean(mean + ng, static_cast<int>(N * G), 1);
    gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);
    gm_f gc2(scratch + 0, 1, 1);
    gm_f gc3(scratch + 1, 1, 1);

    tile_h h0(1, active_d);
    tile_h h1(1, active_d);
    tile_f x_f(1, active_d);
    tile_f dy_f(1, active_d);
    tile_f t0(1, active_d);
    tile_f t1(1, active_d);
    tile_v mean_t(1);
    tile_v rstd_t(1);
    tile_v sum1(1);
    tile_v sum2(1);
    tile_v c2(1);
    tile_v c3(1);

    TLOAD(h0, gx);
    TCVT(x_f, h0);
    TLOAD(h0, gdy);
    TCVT(dy_f, h0);
    TLOAD(mean_t, gmean);
    TLOAD(rstd_t, grstd);

    {
        gm_h gg(gamma + c0, 1, static_cast<int>(C));
        TLOAD(h1, gg);
        TCVT(t0, h1); // gamma
    }

    // sum2 = Σ dy*gamma ; sum1 = Σ dy*gamma*x
    TMUL(t1, dy_f, t0);
    TROWSUM(sum2, t1);
    TMUL(t1, t1, x_f);
    TROWSUM(sum1, t1);

    // c2 = (sum2*mean - sum1) * rstd^3 * s
    TMUL(c2, sum2, mean_t);
    tsub(c2, c2, sum1);
    TMUL(c3, rstd_t, rstd_t);
    TMUL(c3, c3, rstd_t);
    TMUL(c2, c2, c3);
    TMULS(c2, c2, s);

    // c3 = -c2*mean - sum2*rstd*s
    TMUL(c3, c2, mean_t);
    TMULS(c3, c3, -1.0f);
    TMUL(sum1, sum2, rstd_t);
    TMULS(sum1, sum1, s);
    tsub(c3, c3, sum1);

    TSTORE(gc2, c2);
    TSTORE(gc3, c3);
}

// ---------------------------------------------------------------------------
// Stage A2: dX for one (n, g) from spilled c2/c3
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dx_group(dtype *dy, dtype *x, float *rstd, dtype *gamma,
                     float *scratch, dtype *dx, int64_t N, int64_t C,
                     int64_t G, int64_t D, int64_t n, int64_t g) {
    const int64_t ng = n * G + g;
    const int64_t c0 = g * D;
    const int64_t offset = n * C + c0;
    const size_t active_d = static_cast<size_t>(D);

    gm_h gdy(dy + offset, static_cast<int>(N), static_cast<int>(C));
    gm_h gx(x + offset, static_cast<int>(N), static_cast<int>(C));
    gm_h gdx(dx + offset, static_cast<int>(N), static_cast<int>(C));
    gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);
    gm_f gc2(scratch + 0, 1, 1);
    gm_f gc3(scratch + 1, 1, 1);

    tile_h h0(1, active_d);
    tile_h h1(1, active_d);
    tile_f x_f(1, active_d);
    tile_f dy_f(1, active_d);
    tile_f t0(1, active_d);
    tile_f t1(1, active_d);
    tile_v rstd_t(1);
    tile_v c2(1);
    tile_v c3(1);

    TLOAD(h0, gx);
    TCVT(x_f, h0);
    TLOAD(h0, gdy);
    TCVT(dy_f, h0);
    TLOAD(rstd_t, grstd);
    TLOAD(c2, gc2);
    TLOAD(c3, gc3);

    {
        gm_h gg(gamma + c0, 1, static_cast<int>(C));
        TLOAD(h1, gg);
        TCVT(t0, h1); // gamma
    }

    // dX = (rstd*gamma)*dY + c2*X + c3
    TROWEXPANDMUL(t1, t0, rstd_t);
    TMUL(t1, t1, dy_f);
    TROWEXPANDMUL(t0, x_f, c2);
    tadd(t1, t1, t0);
    trowexpandadd(t1, t1, c3);

    TCVT(h0, t1);
    TSTORE(gdx, h0);
}

template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dbeta_group(dtype *dy, dtype *dbeta, int64_t N, int64_t C,
                        int64_t D, int64_t tile_d, int64_t g) {
    const int64_t c0 = g * D;

    for (int64_t d0 = 0; d0 < D; d0 += tile_d) {
        const size_t vd = static_cast<size_t>(
            (d0 + tile_d <= D) ? tile_d : (D - d0));

        tile_h h0(1, vd);
        tile_f dy_f(1, vd);
        tile_f acc(1, vd);
        texpands(acc, 0.0f);

        for (int64_t n = 0; n < N; ++n) {
            gm_h gdy(dy + n * C + c0 + d0, static_cast<int>(N),
                     static_cast<int>(C));
            TLOAD(h0, gdy);
            TCVT(dy_f, h0);
            tadd(acc, acc, dy_f);
        }

        gm_h gdb(dbeta + c0 + d0, 1, static_cast<int>(C));
        TCVT(h0, acc);
        TSTORE(gdb, h0);
    }
}

template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dgamma_group(dtype *dy, dtype *x, float *mean, float *rstd,
                         dtype *dgamma, int64_t N, int64_t C, int64_t G,
                         int64_t D, int64_t tile_d, int64_t g) {
    const int64_t c0 = g * D;

    for (int64_t d0 = 0; d0 < D; d0 += tile_d) {
        const size_t vd = static_cast<size_t>(
            (d0 + tile_d <= D) ? tile_d : (D - d0));

        tile_h h0(1, vd);
        tile_f dy_f(1, vd);
        tile_f x_f(1, vd);
        tile_f t0(1, vd);
        tile_f acc(1, vd);
        tile_v mean_t(1);
        tile_v rstd_t(1);
        texpands(acc, 0.0f);

        for (int64_t n = 0; n < N; ++n) {
            const int64_t ng = n * G + g;
            const int64_t offset = n * C + c0 + d0;

            gm_h gdy(dy + offset, static_cast<int>(N), static_cast<int>(C));
            gm_h gx(x + offset, static_cast<int>(N), static_cast<int>(C));
            gm_f gmean(mean + ng, static_cast<int>(N * G), 1);
            gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);

            TLOAD(h0, gdy);
            TCVT(dy_f, h0);
            TLOAD(h0, gx);
            TCVT(x_f, h0);
            TLOAD(mean_t, gmean);
            TLOAD(rstd_t, grstd);

            TROWEXPANDMUL(t0, x_f, rstd_t);
            TMUL(t0, t0, dy_f);
            TROWEXPANDMUL(x_f, dy_f, mean_t);
            TROWEXPANDMUL(x_f, x_f, rstd_t);
            tsub(t0, t0, x_f);
            tadd(acc, acc, t0);
        }

        gm_h gdg(dgamma + c0 + d0, 1, static_cast<int>(C));
        TCVT(h0, acc);
        TSTORE(gdg, h0);
    }
}

} // namespace gn_grad_1d

template <typename dtype>
void group_norm_grad_1d(dtype *dy, dtype *x, float *mean, float *rstd,
                        dtype *gamma, const int64_t *tiling, dtype *dx,
                        dtype *dgamma, dtype *dbeta) {
    constexpr int64_t tD = 128;

    const int64_t N = tiling[0];
    const int64_t C = tiling[1];
    const int64_t G = tiling[2];
    const int64_t D = C / G;
    const int64_t tile_d = tiling[3] > 0 ? tiling[3] : (D < tD ? D : tD);

    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h =
        Tile<Location::Vec, dtype, 1, tD, BLayout::RowMajor, -1, -1>;
    using tile_f =
        Tile<Location::Vec, float, 1, tD, BLayout::RowMajor, -1, -1>;
    using tile_v = Tile<Location::Vec, float, 1, 32, BLayout::RowMajor, -1, 1>;

    const float s = 1.0f / static_cast<float>(D);
    float scratch[2]; // c2, c3 for one (n,g)

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < G; ++g) {
            gn_grad_1d::fused_params_group<dtype, gm_h, gm_f, tile_h, tile_f,
                                           tile_v>(dy, x, mean, rstd, gamma,
                                                   scratch, N, C, G, D, n, g,
                                                   s);
            gn_grad_1d::dx_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
                dy, x, rstd, gamma, scratch, dx, N, C, G, D, n, g);
        }
    }

    for (int64_t g = 0; g < G; ++g) {
        gn_grad_1d::dbeta_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
            dy, dbeta, N, C, D, tile_d, g);
    }
    for (int64_t g = 0; g < G; ++g) {
        gn_grad_1d::dgamma_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
            dy, x, mean, rstd, dgamma, N, C, G, D, tile_d, g);
    }
}

#endif // SUPERNPU_GROUP_NORM_GRAD_1D_PTO_HPP
