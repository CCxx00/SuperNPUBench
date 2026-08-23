#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>

using namespace pto;

// The tileop API exposes shared TMATMUL operands as SharedTile wrapping a
// TileLeft/TileRight local shape. Q/K/V are all staged into shared tile
// storage via TLOAD, matching kernels/matmul/matmul_shared.hpp.
// 遗留
// 1.高性能上是否存在表达问题？例如，软件pingping流水是否需要暴露(性能)
// 2.layout转换是否需要对程序员可见，数据类型cube-vec之间layout转换

// 4-PE tmatmul FlashAttention programming model.
//
// This kernel models a Blackwell-like execution style where get_thread_idx()
// selects the current PE's output row range. Q/K/V are loaded as full shared
// tiles. The first group TMATMUL consumes Q_shared [kTm,qD] and produces one
// PE-local score slice [kTm/4,kTk] on each PE.
//
// Mathematical semantics:
//   O = softmax((Q * K^T) / sqrt(scaleD)) * V
//   Q: [Sq, qD], K: [Skv, qD], V: [Skv, vD], O: [Sq, vD]
//
// Big-tile vs small-tile naming:
//   - Big tile is the logical tile visible to the collective tmatmul:
//       Q_big: [kTm, qD]
//       K_big: [kTk, qD], consumed by tmatmul as K_big^T [qD, kTk]
//       W_big: [kTm, kTk]
//       V_big: [kTk, vD]
//       O_big/PV_big: [kTm, vD]
//   - Small tile is the PE-local storage unit:
//       W_pe: [kTm/4, kTk]
//       O_pe/PV_pe: [kTm/4, vD]
//   - Q/K/V are shared staging tiles, matching the
//     matmul_shared pattern where both TMATMUL operands live in SharedTile:
//       Q_shared: [kTm, qD]
//       K_shared: [kTk, qD]
//       V_shared: [kTk, vD]
//   - TLOAD uses PEMask=1 so only PE0 issues each shared-tile load. Group
//     TMATMUL maps contiguous kTm/4-row score/output slices to PE0..PE3,
//     matching matmul_shared's C path.
//
// Memory/layout contract:
//   - TLOAD/TSTORE are pure ND DMA copies. They do not transpose, swizzle, or
//     pad data while moving it between global memory and tile storage.
//   - Q, K, V, O global tensors are all RowMajor.
//   - K/V are not split by PE. They are direct row-major shared tiles. TMATMUL
//     consumes K as K^T internally; that interpretation is carried by compute,
//     not by TLOAD.
//   - O is stored as row-major [Sq, vD].
//
// Compute contract:
//   - Each PE independently executes vector tileOPs for its [kTm, *] slice.
//   - Every physical PE-local/shared tile is constrained to at most 8 KiB.
//   - tmatmul is a compiler intrinsic used as a scalar instruction in each PE's
//     program. Each PE passes only its own lhs/acc tile, while K/V are shared
//     staging tiles. The collective execution fuses the PE-local slices into
//     one logical GEMM.
//
// Current simplification:
//   - This TMATMUL example fixes one Q big tile and one K/V big tile per loop
//     step. It intentionally omits the extra array dimensions and merge logic
//     used by multi-block unrolling.

template <typename dtype, int Sq, int Skv, int qD, int vD,
          int kTm, int kTk, int scaleD = qD>
void flash_attention_2d_unroll_shared_impl(dtype *out_ptr, dtype *q_ptr,
                                            dtype *k_ptr, dtype *v_ptr) {
    const uint32_t tid = get_thread_idx();
    constexpr int kPeNum = 4;
    static_assert(kTm % kPeNum == 0,
                  "kTm must be divisible by the PE count");
    constexpr int kPeTm = kTm / kPeNum;

    // This function receives the full Q/O base pointer.
    //   collective Q/O M tile: kTm
    //   current PE C slice   : kPeTm = kTm / 4
    constexpr int kPaddedQ = (qD == 192 ? 256 : qD);

    // Physical-tile size guards removed on request; the TileOP API still
    // enforces the 8 KiB TLOAD Shared limit at compile time.

    // Global tensor layout. All four tensors are RowMajor so TLOAD/TSTORE can
    // remain pure ND-to-ND copies:
    //   gmQ: [Sq,  qD], row stride qD
    //   gmK: [Skv, qD], row stride qD
    //   gmV: [Skv, vD], row stride vD
    //   gmO: [Sq,  vD], row stride vD
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD>>;
    // Reinterpret row-major K[Skv,qD] as the transposed col-major view
    // K^T[qD,Skv], so the right TMATMUL operand has shape [qD,kTk].
    using gmK = global_tensor<dtype, ColMajor<qD, Skv>>;
    using gmV = global_tensor<dtype, RowMajor<Skv, vD>>;
    using gmO = global_tensor<dtype, RowMajor<Sq, vD>>;

    // Q/K/V are shared staging tiles loaded by TLOAD:
    //   tileQ: Q_shared, physical [kTm, kPaddedQ], valid [kTm, qD], NZ-layout
    //   tileK: K^T_shared, physical [kPaddedQ, kTk], valid [qD, kTk], Zn-layout
    //   tileV: V_shared, physical/logical [kTk, vD], Zn-layout
    // The tmatmul intrinsic must read the shared K tile as Zn.
    using tileQLocal = TileLeft<dtype, kTm, kPaddedQ, kTm, qD>;
    using tileQ = SharedTile<tileQLocal>;
    using tileKLocal = TileRight<dtype, kPaddedQ, kTk, qD, kTk>;
    using tileVLocal = TileRight<dtype, kTk, vD>;
    using tileK = SharedTile<tileKLocal>;
    using tileV = SharedTile<tileVLocal>;

    // QK score tiles:
    //   tmatmul input in each PE:
    //     tQ          -> shared Q_big [kTm, qD]
    //     tK          -> shared K tile [kTk, qD]
    //   tmatmul output in each PE:
    //     tW          -> current PE's W_pe [kPeTm, kTk], ordinary Vec tile
    //   logical collective output:
    //     W_big       -> concat W_pe from PE0..PE3, shape [kTm, kTk].
    //
    // tileW/tileWCast are PE-local vector tiles used by online softmax.
    // The group score C has only kPeTm rows. RowMajor keeps the 4-row FP32
    // slice legal because the contiguous kTk columns provide 32-byte alignment.
    using tileW = Tile<Location::Vec, float, kPeTm, kTk, BLayout::RowMajor>;
    using tileWCast = Tile<Location::Vec, dtype,
                           kPeTm, kTk, BLayout::RowMajor>;
    // tilePLeft is the PE-local probability tile converted back to a tmatmul
    // lhs. Like matmul_shared's local C path, every PE describes only its own
    // [kPeTm,kTk] slice; the four PE-local tiles logically form P_big.
    using tilePPadded =
        Tile<Location::Vec, dtype, kPeTm, kTk, BLayout::RowMajor>;
    using tilePLeft = TileLeft<dtype, kPeTm, kTk>;

    // PV/output tiles:
    //   TMATMUL(P_pe, V_shared) -> PV_pe [kPeTm, vD]
    //   current PE receives PV_pe/tileO [kPeTm, vD].
    //   tileO accumulates the online-softmax numerator for this PE row slice.
    //   tileOCast is the dtype tile stored to gmO.
    // P is PE-local while V is shared, so this is the local-A/shared-B
    // TMATMUL form. Its local C has the same row count as the local A:
    // [kPeTm,vD]. No full-kTm physical output container is needed.
    using tileO =
        Tile<Location::Vec, float, kPeTm, vD, BLayout::RowMajor>;
    using tileOCast =
        Tile<Location::Vec, dtype, kPeTm, vD, BLayout::RowMajor>;

    // Online softmax row-state tiles. Each PE owns kPeTm independent query
    // rows, and every row has one scalar max/sum/scale value.
    // Physical cols = 8 only for tile alignment; valid cols = 1.
    //   tileMax/tileSum/tileScale: valid shape [kPeTm, 1]
    using tileMax = Tile<Location::Vec, float, kPeTm, 8, BLayout::RowMajor,
                         kPeTm, 1>;
    using tileSum = Tile<Location::Vec, float, kPeTm, 8, BLayout::RowMajor,
                         kPeTm, 1>;
    using tileScale = Tile<Location::Vec, float, kPeTm, 8, BLayout::RowMajor,
                           kPeTm, 1>;

    using itQ = global_iterator<gmQ, tileQLocal>;
    // global_iterator describes the GM window with the underlying local tile
    // shape. TLOAD may then target either that local tile or its SharedTile
    // wrapper; SharedTile itself is intentionally not an iterator tile type.
    using itK = global_iterator<gmK, tileKLocal>;
    using itV = global_iterator<gmV, tileVLocal>;
    using itO = global_iterator<gmO, tileOCast>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    // Score scaling for softmax(QK / sqrt(scaleD)).
    const float scale = 1.0f / sqrt((float)scaleD);
    // Qb is counted in logical big Q tiles [kTm, qD].
    // Kb is counted in logical big K/V tiles [kTk, qD/vD].
    constexpr int Qb = (Sq + kTm - 1) / kTm;
    constexpr int Kb = (Skv + kTk - 1) / kTk;

#pragma clang loop unroll(full)
    for (int i = 0; i < Qb; ++i) {
        tileQ tQ;

        // Q is staged into shared storage. Only PE0 issues the load
        // (TLOAD<tileQLocal, 1>, PEMask=1); all PEs consume the same shared
        // tile, mirroring how matmul_shared stages A. Q is loaded once per row
        // block and reused across all K/V blocks below.
        //
        // TLOAD remains a direct row-major ND copy and does not change layout.
        auto gQ = gIterQ(i, 0);
        // ND->Nz
        TLOAD<tileQLocal, 1>(tQ, gQ);

        tileMax tMax;
        tileSum tSum;
        tileO tO, tPV;
        tileScale tScale;

        // Initialize online softmax states for the current PE row slice:
        //   tMax valid shape [kTm, 1] = -inf
        //   tSum valid shape [kTm, 1] = 0
        //   tO is initialized after the first PV block.
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);

        // tMax/tSum/tO are ordinary Tile values carried between K blocks.
        // The current backend cannot keep ordinary Tile PHIs across a retained
        // loop, so fully expand this compile-time-bounded online-softmax loop.
#pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
            tileK tK;

            // K storage is row-major [Skv,qD], exposed through the equivalent
            // col-major K^T [qD,Skv] view and loaded as [qD,kTk].
            //
            // For K block j:
            //   before TLOAD:
            //     K^T[0:qD, j*kTk : (j+1)*kTk]
            //   after TLOAD:
            //     tK = K^T_shared, valid shape [qD, kTk].
            // ND->Zn
            auto gK = gIterK(0, j);
            // map to TMATMUL.LD, load tile to staging B
            // 加载到shared tile reg, 只有tid=0会执行加载指令，tid=1，2，3不执行
            TLOAD<tileKLocal, 1>(tK, gK);

            tileW tW;

            // QK group GEMM:
            //   Inputs:
            //     tQ            -> shared Q_big [kTm, qD]
            //     tK            -> shared K tile [kTk, qD], consumed as K^T
            //   Logical output:
            //     W_big = Q_big * K_big^T, shape [kTm, kTk]
            //   Physical output:
            //     tW is the current PE's ordinary W_pe [kPeTm, kTk].
            //
            // Then the current PE scales its own W_pe:
            //   tW = FIXP(Q*K^T) / sqrt(scaleD), shape [kPeTm, kTk].
            // 对应指令gmma
            TMATMUL(tW, tQ, tK);
            TMULS(tW, tW, scale);

            tileMax tNewMax;
            tileSum tNewSum;
            tileWCast tExpW;
            tileMax tLocalMax;
            tileSum tLocalSum;
            tileSum tScaledOldSum;

            // Online softmax is PE-local. The current PE computes row
            // reductions over its own [kPeTm, kTk] score tile.
            //
            //   tW            : current logits, [kPeTm, kTk]
            //   tLocalMax     : rowmax over kTk, [kPeTm, 1]
            //   tNewMax       : max(tMax, tLocalMax), [kPeTm,1]
            //   tScale        : exp(tMax - tNewMax), [kPeTm,1]
            //   tLocalSum     : rowsum(exp(tW - tNewMax)), [kPeTm,1]
            //   tNewSum       : updated denominator, [kPeTm,1]
            TROWMAX(tLocalMax, tW);
            TMAX(tNewMax, tMax, tLocalMax);
            TSUB(tScale, tMax, tNewMax);
            TEXP(tScale, tScale);
            TMUL(tScaledOldSum, tSum, tScale);

            // Convert logits to unnormalized probabilities under tNewMax:
            //   before TROWEXPANDSUB: tW [kPeTm,kTk]
            //   broadcast source    : tNewMax [kPeTm,1]
            //   after TEXP          : tW stores p [kPeTm,kTk]
            //   TROWSUM             : tLocalSum [kPeTm,1]
            //   TCVT                : tExpW [kPeTm,kTk]
            TROWEXPANDSUB(tW, tW, tNewMax);
            TEXP(tW, tW);
            TROWSUM(tLocalSum, tW);
            TCVT(tExpW, tW);
            TADD(tNewSum, tScaledOldSum, tLocalSum);

            tileV tV;

            // V is row-major [Skv, vD] and is loaded once as a full shared
            // [kTk, vD] tile:
            //   before TLOAD:
            //     V[j*kTk : (j+1)*kTk, 0:vD]
            //   after TLOAD:
            //     tV = V_shared, valid shape [kTk, vD].
            auto gV = gIterV(j, 0);
            TLOAD<tileVLocal, 1>(tV, gV);
            // P/probability tile preparation:
            //   tExpW : current PE's Vec probability tile [kPeTm,kTk]
            //   tPLeft: current PE's Left/tmatmul lhs tile [kPeTm,kTk]
            //
            // Across the four independent PE programs, all tPLeft instances
            // logically form P_big [kTm,kTk]. No PE-local P tile is represented
            // as an array here.
            tilePPadded tPPadded;
            tilePLeft tPLeft;

            // PV PE-local GEMM with a shared right operand:
            //   Inputs:
            //     tPLeft        -> current PE's P_pe [kPeTm, kTk]
            //     tV            -> shared V tile [kTk, vD]
            //   Logical output:
            //     PV_big = P_big * V_big, shape [kTm, vD]
            //   Physical output:
            //     tPV is the current PE's ordinary PV_pe [kPeTm, vD].
            // Source and destination are both the current PE's physical
            // [kPeTm,kTk] tile; no full-kTm padding is introduced.
            TINSERT(tPPadded, tExpW, 0, 0);
            TCVT(tPLeft, tPPadded);
            TMATMUL(tPV, tPLeft, tV);

            // Consume the current PV contribution as an ordinary Vec tile.
            //
            // Online output numerator update:
            //   first K/V block:
            //     tO = tPV
            //   later K/V blocks:
            //     tO = tO * exp(m_old - m_new) + tPV
            //
            // TROWEXPANDMUL broadcasts tScale [kPeTm,1] across vD.
            if (j == 0) {
                tO = tPV;
            } else {
                TROWEXPANDMUL(tO, tO, tScale);
                TADD(tO, tO, tPV);
            }

            tMax = tNewMax;
            tSum = tNewSum;
        }

        tileSum tInvSum;
        tileOCast tOCast;

        // Final normalization and store for the current PE:
        //   tInvSum = 1 / tSum, shape [kPeTm,1]
        //   tO *= tInvSum with row broadcast, shape [kPeTm,vD]
        //   tOCast converts float output to dtype, shape [kPeTm,vD]
        //   TSTORE writes the current PE-local O row slice:
        //     O_pe[i*kTm + tid*kPeTm : i*kTm + (tid+1)*kPeTm, 0:vD]
        //
        // Combining stores from all four PEs produces O_big [kTm,vD].
        TRECIP(tInvSum, tSum);
        TROWEXPANDMUL(tO, tO, tInvSum);
        TCVT(tOCast, tO);
        auto dstO = gIterO(i * kPeNum + tid, 0);
        TSTORE(dstO, tOCast);
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_2d_unroll_tmatmul_pto(dtype *out_ptr, dtype *q_ptr,
                                           dtype *k_ptr, dtype *v_ptr) {
    flash_attention_2d_unroll_shared_impl<
        dtype, Sq, Skv, qD, vD, kTm, kTk, scaleD>(
        out_ptr, q_ptr, k_ptr, v_ptr);
}
