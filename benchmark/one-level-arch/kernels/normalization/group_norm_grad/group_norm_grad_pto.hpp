// =============================================================================
// group_norm_grad_pto.hpp — GroupNorm backward, HxW > 1 (one-level PTO)
// =============================================================================
//
// Matches PyTorch GroupNormBackwardKernelImplInternal (group_norm_kernel.cu):
//   1) Spatial reduce  ds/db = Σ_hw (dY*X), Σ_hw dY     per (n,c)
//   2) Fused c2/c3     from ds/db over channels → workspace
//   3) dX              = (rstd*gamma)*dY + c2*X + c3
//   4) dgamma / dbeta  from ds/db over N
//
// Layout: X/dY/dX [N,C,HxW]; mean/rstd [N,G] fp32; gamma/dgamma/dbeta [C]
// workspace float[2*N*C+2*N*G]
// All pointers required (dy,x,mean,rstd,gamma,dx,dgamma,dbeta,workspace).
// tiling[5] = {N, C, G, HxW, tile_hw}
//   tile_hw <= 0 → min(HxW, tCap); spatial R-split when HxW > tile_hw.
//   dX still requires HxW <= tCap (one spatial tile; RF-limited).
//
// tCap=128 (min TSize). Cols=1024 like rms_norm overflows Tile RF here.
// =============================================================================
#ifndef SUPERNPU_GROUP_NORM_GRAD_PTO_HPP
#define SUPERNPU_GROUP_NORM_GRAD_PTO_HPP

#include "normalization/group_norm_grad/group_norm_grad_dyn_ops.hpp"

#include <cstdint>

namespace gn_grad {

using gn_grad_dyn::tadd;
using gn_grad_dyn::texpands;
using gn_grad_dyn::trowexpandadd;
using gn_grad_dyn::tsub;

inline int64_t workspace_elems(int64_t N, int64_t C, int64_t G) {
    return 2 * N * C + 2 * N * G;
}

// ---------------------------------------------------------------------------
// Step 1: spatial reduce for one (n, c) → ds[nc], db[nc]  (HxW R-split)
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void spatial_reduce_nc(dtype *dy, dtype *x, float *ds, float *db,
                              int64_t N, int64_t C, int64_t HxW,
                              int64_t tile_hw, int64_t n, int64_t c) {
    const int64_t nc = n * C + c;
    const int64_t base = nc * HxW;

    gm_f gds(ds + nc, static_cast<int>(N * C), 1);
    gm_f gdb(db + nc, static_cast<int>(N * C), 1);

    tile_v ds_acc(1);
    tile_v db_acc(1);
    tile_v cur(1);
    texpands(ds_acc, 0.0f);
    texpands(db_acc, 0.0f);

    for (int64_t hw0 = 0; hw0 < HxW; hw0 += tile_hw) {
        const size_t vh = static_cast<size_t>(
            (hw0 + tile_hw <= HxW) ? tile_hw : (HxW - hw0));
        const int64_t offset = base + hw0;

        gm_h gdy(dy + offset, static_cast<int>(N * C), static_cast<int>(HxW));
        gm_h gx(x + offset, static_cast<int>(N * C), static_cast<int>(HxW));

        tile_h h0(1, vh);
        tile_f x_f(1, vh);
        tile_f dy_f(1, vh);
        tile_f prod(1, vh);

        TLOAD(h0, gx);
        TCVT(x_f, h0);
        TLOAD(h0, gdy);
        TCVT(dy_f, h0);
        TMUL(prod, dy_f, x_f);
        TROWSUM(cur, prod);
        tadd(ds_acc, ds_acc, cur);
        TROWSUM(cur, dy_f);
        tadd(db_acc, db_acc, cur);
    }

    TSTORE(gds, ds_acc);
    TSTORE(gdb, db_acc);
}

// ---------------------------------------------------------------------------
// Step 2: fused c2/c3 for one (n, g) → c2[ng], c3[ng]
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void fused_params_group(dtype *gamma, float *mean, float *rstd,
                               float *ds, float *db, float *c2_buf,
                               float *c3_buf, int64_t N, int64_t C, int64_t G,
                               int64_t D, int64_t n, int64_t g, float s) {
    const int64_t ng = n * G + g;
    const int64_t c0 = g * D;
    const size_t active_d = static_cast<size_t>(D);

    gm_f gmean(mean + ng, static_cast<int>(N * G), 1);
    gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);
    gm_f gds(ds + n * C + c0, 1, static_cast<int>(C));
    gm_f gdb(db + n * C + c0, 1, static_cast<int>(C));
    gm_f gc2(c2_buf + ng, static_cast<int>(N * G), 1);
    gm_f gc3(c3_buf + ng, static_cast<int>(N * G), 1);

    tile_f ds_f(1, active_d);
    tile_f db_f(1, active_d);
    tile_f gamma_f(1, active_d);
    tile_f t0(1, active_d);
    tile_h h0(1, active_d);
    tile_v mean_t(1);
    tile_v rstd_t(1);
    tile_v sum1(1);
    tile_v sum2(1);
    tile_v c2(1);
    tile_v c3(1);

    TLOAD(ds_f, gds);
    TLOAD(db_f, gdb);
    TLOAD(mean_t, gmean);
    TLOAD(rstd_t, grstd);

    {
        gm_h gg(gamma + c0, 1, static_cast<int>(C));
        TLOAD(h0, gg);
        TCVT(gamma_f, h0);
    }

    TMUL(t0, ds_f, gamma_f);
    TROWSUM(sum1, t0);
    TMUL(t0, db_f, gamma_f);
    TROWSUM(sum2, t0);

    TMUL(c2, sum2, mean_t);
    tsub(c2, c2, sum1);
    TMUL(c3, rstd_t, rstd_t);
    TMUL(c3, c3, rstd_t);
    TMUL(c2, c2, c3);
    TMULS(c2, c2, s);

    TMUL(c3, c2, mean_t);
    TMULS(c3, c3, -1.0f);
    TMUL(sum1, sum2, rstd_t);
    TMULS(sum1, sum1, s);
    tsub(c3, c3, sum1);

    TSTORE(gc2, c2);
    TSTORE(gc3, c3);
}

// ---------------------------------------------------------------------------
// Step 3: dX for one (n, c) using stored c2/c3 and rstd*gamma
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dx_nc(dtype *dy, dtype *x, dtype *gamma, float *rstd, float *c2_buf,
                  float *c3_buf, dtype *dx, int64_t N, int64_t C, int64_t G,
                  int64_t D, int64_t HxW, int64_t n, int64_t c) {
    const int64_t g = c / D;
    const int64_t ng = n * G + g;
    const int64_t offset = (n * C + c) * HxW;
    const size_t active_hw = static_cast<size_t>(HxW);

    gm_h gdy(dy + offset, static_cast<int>(N * C), static_cast<int>(HxW));
    gm_h gx(x + offset, static_cast<int>(N * C), static_cast<int>(HxW));
    gm_h gdx(dx + offset, static_cast<int>(N * C), static_cast<int>(HxW));
    gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);
    gm_f gc2(c2_buf + ng, static_cast<int>(N * G), 1);
    gm_f gc3(c3_buf + ng, static_cast<int>(N * G), 1);

    tile_h h0(1, active_hw);
    tile_f x_f(1, active_hw);
    tile_f dy_f(1, active_hw);
    tile_f dx_f(1, active_hw);
    tile_f tmp(1, active_hw);
    tile_v rstd_t(1);
    tile_v c1(1);
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
        gm_h gg(gamma + c, 1, 1);
        tile_h hg(1, 1);
        tile_v gv(1);
        TLOAD(hg, gg);
        TCVT(gv, hg);
        TMUL(c1, gv, rstd_t);
    }

    TROWEXPANDMUL(dx_f, dy_f, c1);
    TROWEXPANDMUL(tmp, x_f, c2);
    tadd(dx_f, dx_f, tmp);
    trowexpandadd(dx_f, dx_f, c3);

    TCVT(h0, dx_f);
    TSTORE(gdx, h0);
}

// ---------------------------------------------------------------------------
// Step 4a: dbeta
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f>
inline void dbeta_group(float *db, dtype *dbeta, int64_t N, int64_t C,
                        int64_t D, int64_t g) {
    const int64_t c0 = g * D;
    const size_t active_d = static_cast<size_t>(D);

    tile_f acc(1, active_d);
    tile_f cur(1, active_d);
    tile_h h0(1, active_d);
    texpands(acc, 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        gm_f gdb(db + n * C + c0, 1, static_cast<int>(C));
        TLOAD(cur, gdb);
        tadd(acc, acc, cur);
    }

    gm_h gout(dbeta + c0, 1, static_cast<int>(C));
    TCVT(h0, acc);
    TSTORE(gout, h0);
}

// ---------------------------------------------------------------------------
// Step 4b: dgamma
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dgamma_group(float *ds, float *db, float *mean, float *rstd,
                         dtype *dgamma, int64_t N, int64_t C, int64_t G,
                         int64_t D, int64_t g) {
    const int64_t c0 = g * D;
    const size_t active_d = static_cast<size_t>(D);

    tile_f acc(1, active_d);
    tile_f ds_f(1, active_d);
    tile_f db_f(1, active_d);
    tile_f t0(1, active_d);
    tile_h h0(1, active_d);
    tile_v mean_t(1);
    tile_v rstd_t(1);
    texpands(acc, 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        const int64_t ng = n * G + g;
        gm_f gds(ds + n * C + c0, 1, static_cast<int>(C));
        gm_f gdb(db + n * C + c0, 1, static_cast<int>(C));
        gm_f gmean(mean + ng, static_cast<int>(N * G), 1);
        gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);

        TLOAD(ds_f, gds);
        TLOAD(db_f, gdb);
        TLOAD(mean_t, gmean);
        TLOAD(rstd_t, grstd);

        TROWEXPANDMUL(t0, db_f, mean_t);
        tsub(t0, ds_f, t0);
        TROWEXPANDMUL(t0, t0, rstd_t);
        tadd(acc, acc, t0);
    }

    gm_h gout(dgamma + c0, 1, static_cast<int>(C));
    TCVT(h0, acc);
    TSTORE(gout, h0);
}

} // namespace gn_grad

// tiling: [N, C, G, HxW, tile_hw]
// workspace: float[2*N*C + 2*N*G]
template <typename dtype>
void group_norm_grad(dtype *dy, dtype *x, float *mean, float *rstd,
                     dtype *gamma, const int64_t *tiling, dtype *dx,
                     dtype *dgamma, dtype *dbeta, float *workspace) {
    constexpr int64_t tCap = 128;

    const int64_t N = tiling[0];
    const int64_t C = tiling[1];
    const int64_t G = tiling[2];
    const int64_t HxW = tiling[3];
    const int64_t tile_hw =
        tiling[4] > 0 ? tiling[4] : (HxW < tCap ? HxW : tCap);
    const int64_t D = C / G;

    float *ds = workspace;
    float *db = workspace + N * C;
    float *c2_buf = workspace + 2 * N * C;
    float *c3_buf = c2_buf + N * G;

    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h =
        Tile<Location::Vec, dtype, 1, tCap, BLayout::RowMajor, -1, -1>;
    using tile_f =
        Tile<Location::Vec, float, 1, tCap, BLayout::RowMajor, -1, -1>;
    using tile_v = Tile<Location::Vec, float, 1, 32, BLayout::RowMajor, -1, 1>;

    const float s = 1.0f / static_cast<float>(D * HxW);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            gn_grad::spatial_reduce_nc<dtype, gm_h, gm_f, tile_h, tile_f,
                                       tile_v>(dy, x, ds, db, N, C, HxW,
                                               tile_hw, n, c);
        }
    }

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < G; ++g) {
            gn_grad::fused_params_group<dtype, gm_h, gm_f, tile_h, tile_f,
                                        tile_v>(gamma, mean, rstd, ds, db,
                                                c2_buf, c3_buf, N, C, G, D, n,
                                                g, s);
        }
    }
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            gn_grad::dx_nc<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
                dy, x, gamma, rstd, c2_buf, c3_buf, dx, N, C, G, D, HxW, n, c);
        }
    }

    for (int64_t g = 0; g < G; ++g) {
        gn_grad::dbeta_group<dtype, gm_h, gm_f, tile_h, tile_f>(db, dbeta, N,
                                                                C, D, g);
    }
    for (int64_t g = 0; g < G; ++g) {
        gn_grad::dgamma_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
            ds, db, mean, rstd, dgamma, N, C, G, D, g);
    }
}

#endif // SUPERNPU_GROUP_NORM_GRAD_PTO_HPP
