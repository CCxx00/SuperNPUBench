// =============================================================================
// rms_norm_binary_pto.hpp — RMSNorm for g_r > tile_r (R-split)
// =============================================================================
//
// Shape dims: A (outer / row), R (reduce / col). tiling = {g_a, g_r, tile_a, tile_r}.
//
// Formula:
//   out[a] = x[a] * rsqrt(mean(x[a]^2) + eps)
//          = x[a] * rsqrt( (Σ_j x[a,j]^2) / g_r + eps )
//
// R 被切成 Rb = ceil(g_r / tile_r) 个 tile（默认 tile_r=1024）。
// 每个 tile 先做局部 TROWSUM(x^2)，再跨 tile 得到整行 Σx^2。
//
// Pass1：循环外把 sum 置零，循环内统一 TADD(sum, sum, cur)，不是 GetCacheId 进位合并。
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_BINARY_PTO_HPP
#define SUPERNPU_RMS_NORM_BINARY_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_bin {

// -----------------------------------------------------------------------------
// 常量
//   kVecCols: 计算用 tile_v 物理 Cols（与 rms_norm_pto 一致，TSize 合法）
// -----------------------------------------------------------------------------
constexpr int kVecCols = 32;

// 累加器初值：与 tile_v 同宽的全 0（循环外 TLOAD 进 sum）
alignas(256) inline float kZeroTile[kVecCols] = {};

inline int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

// -----------------------------------------------------------------------------
// Newton rsqrt：out ≈ 1/sqrt(a)
//   输入 a = mean(x^2)+eps，输出供 TROWEXPANDMUL 广播乘到整行。
// -----------------------------------------------------------------------------
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

// =============================================================================
// Entry: rms_norm_binary
//   tiling[4] = {g_a, g_r, tile_a, tile_r}
//   workspace : 保留参数（当前未使用）
// =============================================================================
template <typename dtype>
void rms_norm_binary(dtype *x, const int64_t *tiling, dtype *out,
                     float *workspace, float eps = 1e-6f) {
    (void)workspace;

    // -------------------------------------------------------------------------
    // §0 解析 tiling / 派生量
    //   tA/tR   : Tile 模板物理尺寸上限（编译期；A=row, R=col）
    //   gA/gR   : 全局 shape
    //   tile_a/r: 本次切块；Rb = 沿 R 方向的 tile 个数
    //   inv_r   : 1/g_r，后面做 mean(x^2)
    // -------------------------------------------------------------------------
    constexpr int64_t tA = 1;
    constexpr int64_t tR = 1024;

    const int64_t gA = tiling[0];
    const int64_t gR = tiling[1];
    const int64_t tile_a = tiling[2] > 0 ? tiling[2] : tA;
    const int64_t tile_r = tiling[3] > 0 ? tiling[3] : tR;
    const int64_t Rb = (gR + tile_r - 1) / tile_r;
    const float inv_r = 1.0f / static_cast<float>(gR);

    // -------------------------------------------------------------------------
    // §1 Tile / GM 类型
    //   tile_h : fp16 输入/输出块，Valid 动态（尾块 active_r）
    //   tile_f : fp32 计算块，与 tile_h 同 Valid
    //   tile_v : 行归约后的 1×ValidCol 向量（Cols=32, Valid=1,1）
    // -------------------------------------------------------------------------
    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_v = Tile<Location::Vec, float, tA, rms_bin::kVecCols,
                        BLayout::RowMajor, 1, 1>;

    // -------------------------------------------------------------------------
    // §2 按 A 维处理（当前实现假定 tile_a==1，故外层一次一行）
    // -------------------------------------------------------------------------
    for (int64_t ia = 0; ia < gA; ++ia) {
        constexpr size_t active_a = 1;

        // cur  : 当前 R-tile 的 Σx^2（TROWSUM 结果）
        // sum  : 跨 tile 累加后的整行 Σx^2（循环外置 0）
        // mean : sum / g_r
        // denom: mean + eps
        // rms  : rsqrt(denom)
        tile_v cur, sum, mean, denom, rms;

        // =====================================================================
        // Pass1 — 跨 R-tile 归约 Σx^2
        //
        // 循环外：TLOAD 全 0 进 sum
        // 循环内：对每个 tr，TROWSUM → TADD(sum, sum, cur)（无分支）
        // 结束后：sum == Σ_{j=0}^{g_r-1} x[ia,j]^2
        // =====================================================================
        gm_f gz(rms_bin::kZeroTile, 1, rms_bin::kVecCols);
        TLOAD(sum, gz);

        for (int64_t tr = 0; tr < Rb; ++tr) {
            const int64_t active_r =
                rms_bin::min64(tile_r, gR - tr * tile_r);
            const int64_t offset = ia * gR + tr * tile_r;
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));

            tile_h src_h(active_a, static_cast<size_t>(active_r));
            tile_f src(active_a, static_cast<size_t>(active_r));
            tile_f sq(active_a, static_cast<size_t>(active_r));

            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TMUL(sq, src, src);
            TROWSUM(cur, sq);
            TADD(sum, sum, cur);
        }

        // =====================================================================
        // Pass1.5 — 由 Σx^2 得到行缩放因子 rms
        //   mean  = sum * (1/g_r)
        //   denom = mean + eps
        //   rms   ≈ 1/sqrt(denom)
        // =====================================================================
        TMULS(mean, sum, inv_r);
        TADDS(denom, mean, eps);
        rms_bin::rsqrt_newton(rms, denom);

        // =====================================================================
        // Pass2 — 写回输出
        //   再次按 R-tile 遍历：out_tile = x_tile * rms（行广播乘）
        //   fp32→fp16 后 TSTORE
        // =====================================================================
        for (int64_t tr = 0; tr < Rb; ++tr) {
            const int64_t active_r =
                rms_bin::min64(tile_r, gR - tr * tile_r);
            const int64_t offset = ia * gR + tr * tile_r;
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t go(out + offset, static_cast<int>(gA), static_cast<int>(gR));

            tile_h src_h(active_a, static_cast<size_t>(active_r));
            tile_h dst_h(active_a, static_cast<size_t>(active_r));
            tile_f src(active_a, static_cast<size_t>(active_r));
            tile_f dst(active_a, static_cast<size_t>(active_r));

            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TROWEXPANDMUL(dst, src, rms);
            TCVT(dst_h, dst);
            TSTORE(go, dst_h);
        }
    }
}

#endif // SUPERNPU_RMS_NORM_BINARY_PTO_HPP
