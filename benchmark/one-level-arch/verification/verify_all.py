#!/usr/bin/env python3
"""
SuperNPUBench — CPU PyTorch verification for one-level-arch kernels.

Each function generates (input_tensors, expected_output_tensors) for one kernel.
Usage:
    python3 verify_all.py                       # run all checks
    python3 verify_all.py --kernel matmul        # run one
    python3 verify_all.py --kernel matmul --print # print tensors

Reference data is deterministic (fixed seed) and can be compared against
ELF simulation output buffers dumped by gfrun/gfsim.
"""

import argparse
import math
import struct
import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_np(t):
    """torch tensor → numpy row-major array."""
    return t.detach().cpu().numpy()

def gen_seeds(n, seed=SEED):
    return torch.manual_seed(seed) if n is None else torch.manual_seed(seed + n)

def randn(shape, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype)

def randint(shape, low=0, high=100, dtype=torch.int32, seed=0):
    torch.manual_seed(seed)
    return torch.randint(low, high, tuple(shape), dtype=dtype)

def write_bin(path, arr):
    """Write numpy array as raw binary (matching C memory layout)."""
    arr.tofile(path)

def read_bin(path, dtype, count):
    """Read raw binary into numpy array."""
    return np.fromfile(path, dtype=dtype, count=count)

def compare(a, b, name, rtol=1e-3, atol=1e-4):
    """Compare two numpy arrays, return True if close."""
    if a.shape != b.shape:
        print(f"  ✗ {name}: shape mismatch {a.shape} vs {b.shape}")
        return False
    if np.issubdtype(a.dtype, np.floating):
        close = np.allclose(a, b, rtol=rtol, atol=atol)
    else:
        close = np.array_equal(a, b)
    if close:
        print(f"  ✓ {name}: PASS (shape={a.shape}, max_err={np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))) if a.dtype != np.bool_ else 0})")
    else:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print(f"  ✗ {name}: FAIL (shape={a.shape}, max_err={np.max(diff):.6f}, mean_err={np.mean(diff):.6f})")
    return close


# ===========================================================================
# 1. Matmul: C = A @ B
# ===========================================================================

def verify_matmul(M=256, N=256, K=256, dtype=torch.float16):
    """C = A @ B, A:[M,K], B:[K,N], C:[M,N] float32."""
    A = randn((M, K), dtype=dtype, seed=1)
    B = randn((K, N), dtype=dtype, seed=2)
    C_ref = (A.float() @ B.float())  # fp32 accumulation
    return {"A": A, "B": B}, {"C": C_ref.float()}


# ===========================================================================
# 2. Flash Attention (dense): O = softmax(Q@K^T / sqrt(d)) @ V
# ===========================================================================

def verify_fa_2d_unroll(Sq=256, Skv=512, qD=128, vD=128, dtype=torch.float16):
    Q = randn((Sq, qD), dtype=dtype, seed=1)
    K = randn((Skv, qD), dtype=dtype, seed=2)
    V = randn((Skv, vD), dtype=dtype, seed=3)
    scale = 1.0 / math.sqrt(qD)
    scores = (Q.float() @ K.T.float()) * scale          # [Sq, Skv]
    attn = F.softmax(scores, dim=-1)                      # [Sq, Skv]
    O = attn @ V.float()                                  # [Sq, vD]
    return {"Q": Q, "K": K, "V": V}, {"O": O.to(dtype).float()}


# ===========================================================================
# 3. Sparse Flash Attention (block-sparse CSR)
# ===========================================================================

def verify_sfa(Sq=256, Skv=512, qD=128, vD=128, kTm=16, kTk=32, dtype=torch.float16):
    Q = randn((Sq, qD), dtype=dtype, seed=1)
    K = randn((Skv, qD), dtype=dtype, seed=2)
    V = randn((Skv, vD), dtype=dtype, seed=3)
    scale = 1.0 / math.sqrt(qD)

    Qb = Sq // kTm
    Kb = Skv // kTk
    window = 2  # local window attention

    kv_idx = []
    kv_off = [0]
    for i in range(Qb):
        lo = max(0, i - window)
        hi = min(Kb, i + window + 1)
        for j in range(lo, hi):
            kv_idx.append(j)
        kv_off.append(len(kv_idx))

    kv_idx = torch.tensor(kv_idx, dtype=torch.int32)
    kv_off = torch.tensor(kv_off, dtype=torch.int32)

    O = torch.zeros(Sq, vD, dtype=torch.float32)
    for i in range(Qb):
        act = kv_idx[kv_off[i]:kv_off[i+1]].long()
        q_block = Q[i*kTm:(i+1)*kTm].float()
        k_block = K[act[0].item()*kTk : (act[-1].item()+1)*kTk].float()
        v_block = V[act[0].item()*kTk : (act[-1].item()+1)*kTk].float()

        active_mask = torch.zeros(Kb, dtype=torch.bool)
        active_mask[act] = True
        k_sel = K[act[0].item()*kTk : (act[-1].item()+1)*kTk].float()
        v_sel = V[act[0].item()*kTk : (act[-1].item()+1)*kTk].float()

        s = q_block @ k_sel.T * scale
        p = F.softmax(s, dim=-1)
        O[i*kTm:(i+1)*kTm] = p @ v_sel

    return {"Q": Q, "K": K, "V": V, "kv_idx": kv_idx, "kv_off": kv_off}, {"O": O}


# ===========================================================================
# 4. FA Softmax (standalone): out = softmax(score, dim=-1)
# ===========================================================================

def verify_fa_softmax(Sq=256, Skv=512):
    score = randn((Sq, Skv), dtype=torch.float32, seed=1)
    out = F.softmax(score, dim=-1)
    return {"score": score}, {"out": out}


# ===========================================================================
# 5. Transpose (2D)
# ===========================================================================

def verify_transpose_2d(Rows=256, Cols=256, dtype=torch.float16):
    inp = randn((Rows, Cols), dtype=dtype, seed=1)
    out = inp.T.contiguous()
    return {"input": inp}, {"output": out}


# ===========================================================================
# 6. Transpose (N-D)
# ===========================================================================

def verify_transpose_nd(shape=(1, 8, 4096, 3), axis0=2, axis1=3, dtype=torch.float16):
    inp = randn(shape, dtype=dtype, seed=1)
    out = inp.transpose(axis0, axis1).contiguous()
    return {"input": inp, "in_shape": torch.tensor(shape),
            "out_shape": torch.tensor(out.shape)}, {"output": out}


# ===========================================================================
# 7. Reduction: column sum  out[1,N] = sum(in[M,N], dim=0)
# ===========================================================================

def verify_reducesum_col(M=2048, N=64, dtype=torch.float16):
    inp = randn((M, N), dtype=dtype, seed=1)
    out = inp.float().sum(dim=0, keepdim=True)
    return {"in": inp}, {"out": out}


def verify_reducesum_row(M=16, N=8192, dtype=torch.int32):
    inp = randint((M, N), low=-100, high=100, dtype=dtype, seed=1)
    out = inp.sum(dim=1, keepdim=True)
    return {"in": inp}, {"out": out}


def verify_reducemax_col(M=2048, N=64, dtype=torch.int32):
    inp = randint((M, N), low=-100, high=100, dtype=dtype, seed=1)
    out = inp.max(dim=0, keepdim=True)[0]
    return {"in": inp}, {"out": out}


def verify_reducemax_row(M=16, N=8192, dtype=torch.int32):
    inp = randint((M, N), low=-100, high=100, dtype=dtype, seed=1)
    out = inp.max(dim=1, keepdim=True)[0]
    return {"in": inp}, {"out": out}


# ===========================================================================
# 8. GELU (custom polynomial)
# ===========================================================================

def verify_gelu(M=196608, dtype=torch.bfloat16):
    """GELU via custom polynomial (not PyTorch tanh-gelu)."""
    x = randn((1, M), dtype=dtype, seed=1)
    xf = x.float()
    t = xf.clamp(-5.75, 5.75)
    t2 = t * t

    # Custom coefficients
    AM1 = -1.596
    A0  = -7.267e-2
    A1  = 6.519e-5
    A2  = 1.106e-4
    A3  = -7.929e-6
    A4  = 2.645e-7
    A5  = -3.512e-9

    p = ((((((A5*t2 + A4)*t2 + A3)*t2 + A2)*t2 + A1)*t2 + A0)*t2 + AM1)
    y = xf / (1.0 + torch.exp(t * p))
    return {"in": x}, {"out": y.to(dtype).float()}


# ===========================================================================
# 9. Broadcast
# ===========================================================================

def verify_broadcast(in_shape=(1334, 1), out_shape=(1334, 129), dtype=torch.float16):
    inp = randn(in_shape, dtype=dtype, seed=1)
    out = inp.broadcast_to(out_shape).contiguous()
    return {"in": inp, "in_shape": torch.tensor(in_shape),
            "out_shape": torch.tensor(out_shape)}, {"out": out.reshape(-1)}


# ===========================================================================
# 10. Concat (gather mode)
# ===========================================================================

def verify_concat_gather(N_tables=1000, Rows=64, Cols=2, dtype=torch.int32):
    """N tables of (Rows, Cols) → concat along dim=1 → (Rows, N*Cols)."""
    tables = randint((N_tables, Rows, Cols), low=0, high=1000, dtype=dtype, seed=1)
    out = tables.permute(1, 0, 2).reshape(Rows, N_tables * Cols)
    in_flat = tables.reshape(-1)
    out_flat = out.reshape(-1)
    in_shape = torch.tensor([Rows, Cols])
    out_shape = torch.tensor([Rows, N_tables * Cols])
    return {"in": in_flat, "in_shape": in_shape, "out_shape": out_shape}, {"out": out_flat}


# ===========================================================================
# 11. Gather (row-index)
# ===========================================================================

def verify_gather(gK=131072, gM=32, gN=256, dtype=torch.float32):
    table = randn((gK, gN), dtype=dtype, seed=1)
    indices = randint((gM,), low=0, high=gK, dtype=torch.int32, seed=2)
    out = table[indices.long()]  # [gM, gN]
    return {"table": table, "indices": indices}, {"out": out}


# ===========================================================================
# 12. DeepSeek: fused_weight  out = a * b (element-wise)
# ===========================================================================

def verify_fused_weight(HcMult=2, Hidden=64, dtype=torch.bfloat16):
    a = randn((HcMult, Hidden), dtype=dtype, seed=1)
    b = randn((HcMult, Hidden), dtype=dtype, seed=2)
    out = (a.float() * b.float())
    return {"weight_hidden": a, "weight_embed": b}, {"weight_fused": out}


# ===========================================================================
# 13. DeepSeek: rms_norm  out = x * rsqrt(mean(x^2) + eps)
# ===========================================================================

def verify_rms_norm(M=16, N=8, eps=1e-6):
    x = randn((M, N), dtype=torch.float32, seed=1)
    ms = (x ** 2).mean(dim=1, keepdim=True)
    out = x * torch.rsqrt(ms + eps)
    return {"x": x}, {"out": out}


# ===========================================================================
# 14. DeepSeek: batched_transpose  out[b] = in[b].T
# ===========================================================================

def verify_batched_transpose(Batch=2, Rows=16, Cols=16, dtype=torch.float32):
    inp = randn((Batch, Rows, Cols), dtype=dtype, seed=1)
    out = inp.transpose(1, 2).contiguous()
    return {"input": inp.reshape(-1)}, {"output": out.reshape(-1)}


# ===========================================================================
# 15. DeepSeek: expand_to_mhc_fwd  o[i,m,j] = x[i,j]
# ===========================================================================

def verify_expand_to_mhc_fwd(NumTokens=16, Hidden=64, MhcMult=4, dtype=torch.bfloat16):
    x = randn((NumTokens, Hidden), dtype=dtype, seed=1)
    out = x.unsqueeze(1).expand(-1, MhcMult, -1).reshape(NumTokens * MhcMult, Hidden)
    return {"x": x}, {"o": out}


# ===========================================================================
# 16. DeepSeek: expand_to_mhc_bwd  x_grad = sum(o_grad, dim=mhc)
# ===========================================================================

def verify_expand_to_mhc_bwd(NumTokens=16, Hidden=64, MhcMult=16, dtype=torch.bfloat16):
    o_grad = randn((NumTokens * MhcMult, Hidden), dtype=dtype, seed=1)
    x_grad = o_grad.view(NumTokens, MhcMult, Hidden).sum(dim=1)
    return {"o_grad": o_grad}, {"x_grad": x_grad.to(dtype)}


# ===========================================================================
# 17. DeepSeek: topk_gate  greedy top-k with tie-break by smaller index
# ===========================================================================

def verify_topk_gate(NumTokens=16, NumExperts=32, NumTopk=4):
    scores = randn((NumTokens, NumExperts), dtype=torch.float32, seed=1)
    s = scores.clone()
    topk_idx = torch.zeros(NumTokens, NumTopk, dtype=torch.int32)
    for k in range(NumTopk):
        m = s.max(dim=1, keepdim=True)[0]
        mask = (s == m)
        idx = mask.int().argmax(dim=1)
        topk_idx[:, k] = idx
        s[torch.arange(NumTokens), idx] = float('-inf')
    return {"scores": scores}, {"topk_idx": topk_idx}


# ===========================================================================
# 18. DeepSeek: normalize_weight  out = w / (sum(w) + eps)
# ===========================================================================

def verify_normalize_weight(NumTokens=16, NumTopk=8, eps=1e-20):
    w = randn((NumTokens, NumTopk), dtype=torch.float32, seed=1).abs()
    s = w.sum(dim=1, keepdim=True) + eps
    out = w / s
    return {"topk_weights": w}, {"denominator": s, "normalized_weights": out}


# ===========================================================================
# 19. DeepSeek: group_count  out[g] = count(group_idx == g)
# ===========================================================================

def verify_group_count(NumTokens=16, NumTopk=8, NumGroups=32):
    group_idx = randint((NumTokens, NumTopk), low=0, high=NumGroups, dtype=torch.int32, seed=1)
    out = torch.bincount(group_idx.flatten(), minlength=NumGroups).view(1, NumGroups)
    return {"group_idx": group_idx}, {"out": out}


# ===========================================================================
# 20. DeepSeek: aux_fi  out[e] = cnt[e] * NumExperts / (NumTokens * num_aux_topk)
# ===========================================================================

def verify_aux_fi(NumTokens=16, NumTopk=8, NumExperts=32, num_aux_topk=8):
    topk_idx = randint((NumTokens, NumTopk), low=0, high=NumExperts, dtype=torch.int32, seed=1)
    cnt = torch.bincount(topk_idx.flatten(), minlength=NumExperts).float()
    out = cnt * NumExperts / (NumTokens * num_aux_topk)
    return {"topk_idx": topk_idx}, {"out": out.view(1, -1)}


# ===========================================================================
# 21. DeepSeek: cast_back_per_token  out = x * sf (per-row)
# ===========================================================================

def verify_cast_back_per_token(M=16, K=16, dtype=torch.bfloat16):
    x = randn((M, K), dtype=dtype, seed=1)
    sf = randn((M, 1), dtype=torch.float32, seed=2).abs()
    out = x.float() * sf
    return {"x": x, "sf": sf}, {"out": out}


def verify_cast_back_per_channel(M=16, K=32, dtype=torch.bfloat16):
    x = randn((M, K), dtype=dtype, seed=1)
    sf = randn((1, K), dtype=torch.float32, seed=2).abs()
    out = x.float() * sf
    return {"x": x, "sf": sf}, {"out": out}


# ===========================================================================
# 22. DeepSeek: per_token_cast  (quantize with per-row scale)
# ===========================================================================

def verify_per_token_cast(M=16, Npc=16, max_val=448.0, clamp_min=1e-6, dtype=torch.bfloat16):
    x = randn((M, Npc), dtype=dtype, seed=1)
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(clamp_min)
    sf = amax / max_val
    out = (xf * (max_val / amax)).to(dtype)
    return {"x": x}, {"out_sf": sf, "out": out}


# ===========================================================================
# 23. DeepSeek: per_channel_cast  (quantize with per-col scale)
# ===========================================================================

def verify_per_channel_cast(Npt=16, Hidden=32, max_val=448.0, clamp_min=1e-6, dtype=torch.bfloat16):
    x = randn((Npt, Hidden), dtype=dtype, seed=1)
    xf = x.float()
    amax = xf.abs().amax(dim=0, keepdim=True).clamp_min(clamp_min)
    sf = amax / max_val
    out = (xf * (max_val / amax)).to(dtype)
    return {"x": x}, {"out_sf": sf, "out": out}


# ===========================================================================
# 24. DeepSeek: swiglu_forward_and_per_token_cast
# ===========================================================================

def verify_swiglu(M=16, Hidden=16, Npc=16, max_val=448.0, clamp_min=1e-6, dtype=torch.bfloat16):
    x = randn((M, 2 * Hidden), dtype=dtype, seed=1)
    g = x[:, :Hidden].float()
    u = x[:, Hidden:].float()
    silu = F.silu(g)
    sw = silu * u
    sw_grouped = sw.view(M, -1, Npc)
    amax = sw_grouped.abs().amax(dim=-1, keepdim=True).clamp_min(clamp_min)
    sf = amax / max_val
    out = (sw_grouped * (max_val / amax)).to(dtype).view(M, Hidden)
    return {"x": x}, {"out_sf": sf, "out": out}


# ===========================================================================
# 25. DeepSeek: reduce_fused  out = sum(x[pos] * w)
# ===========================================================================

def verify_reduce_fused(NumTokens=16, Hidden=64, NumTopk=4, NumExpanded=32, dtype=torch.bfloat16):
    x = randn((NumExpanded, Hidden), dtype=dtype, seed=1)
    w = randn((NumTokens, NumTopk), dtype=torch.float32, seed=2).abs()
    pos = randint((NumTokens, NumTopk), low=0, high=NumExpanded, dtype=torch.int32, seed=3)
    out = torch.zeros(NumTokens, Hidden, dtype=torch.float32)
    for n in range(NumTokens):
        for k in range(NumTopk):
            p = pos[n, k].item()
            out[n] += x[p].float() * w[n, k]
    return {"x": x, "topk_weights": w, "token_topk_to_pos": pos}, {"out": out}


# ===========================================================================
# 26. DeepSeek: inplace_unique_group_indices
# ===========================================================================

def verify_inplace_unique(NumTokens=16, NumTopk=8):
    indices = randint((NumTokens, NumTopk), low=0, high=8, dtype=torch.int32, seed=1)
    out = indices.clone()
    for t in range(NumTokens):
        seen = set()
        for k in range(NumTopk):
            v = out[t, k].item()
            if v in seen:
                out[t, k] = -1
            else:
                seen.add(v)
    return {"indices": indices}, {"out": out}


# ===========================================================================
# 27. DeepSeek: sinkhorn  (doubly-stochastic normalization)
# ===========================================================================

def verify_sinkhorn(NumTokens=2, Hidden=16, Repeat=1, eps=1e-20):
    x = randn((NumTokens * Hidden, Hidden), dtype=torch.float32, seed=1)
    out = x.view(NumTokens, Hidden, Hidden).clone()
    for _ in range(Repeat):
        m = out.max(dim=-1, keepdim=True)[0]
        out = torch.exp(out - m)
        s = out.sum(dim=-1, keepdim=True)
        out = out / (s + eps) + eps
        cs = out.sum(dim=-2, keepdim=True)
        out = out / (cs + eps)
    return {"comb_res_mix": x}, {"comb_res_mix_out": out.reshape(-1)}


# ===========================================================================
# 28. DeepSeek: fn_normw_merge_fwd  out = fn * normw (col-broadcast)
# ===========================================================================

def verify_fn_normw_merge(M=16, N=32):
    fn = randn((M, N), dtype=torch.float32, seed=1)
    normw = randn((1, N), dtype=torch.float32, seed=2)
    out = fn * normw
    return {"fn": fn, "normw": normw}, {"out_fn": out}


# ===========================================================================
# 29. DeepSeek: mask_indices_by_tp
# ===========================================================================

def verify_mask_indices_by_tp(NumTokens=16, NumTopk=8, per_gpu=1, per_dp=1, num_tp_ranks=1, tp_rank=0):
    indices = randint((NumTokens, NumTopk), low=0, high=32, dtype=torch.int32, seed=1)
    out = indices.clone()
    q = out // per_gpu
    r = q % num_tp_ranks
    is_rank = (r == tp_rank)
    local = out - tp_rank * per_gpu
    dp = local // per_dp
    local = local - dp * (per_dp - per_gpu)
    is_ge0 = (local >= 0)
    result = torch.where(is_rank & is_ge0, local, torch.full_like(out, -1))
    return {"indices": indices}, {"out": result}


# ===========================================================================
# 30. DeepSeek: engram_hash_layer
# ===========================================================================

def verify_engram_hash(NumTokens=16, MaxNgramSize=8, NumEmbedPerNgram=8):
    ngram_ids = randint((NumTokens, MaxNgramSize), low=1, high=1000, dtype=torch.int32, seed=1)
    multipliers = randint((MaxNgramSize,), low=1, high=100, dtype=torch.int64, seed=2)
    kNumOutCols = (MaxNgramSize - 1) * NumEmbedPerNgram
    vocab_sizes = randint((kNumOutCols,), low=10, high=500, dtype=torch.int32, seed=3)
    offsets = randint((kNumOutCols,), low=0, high=100, dtype=torch.int32, seed=4)

    output = torch.zeros(NumTokens, kNumOutCols, dtype=torch.int32)
    for t in range(NumTokens):
        hash_val = 0
        for n in range(MaxNgramSize):
            hash_val ^= int(ngram_ids[t, n].item()) * int(multipliers[n].item())
            if n > 0:
                for j in range(NumEmbedPerNgram):
                    col = (n - 1) * NumEmbedPerNgram + j
                    output[t, col] = (hash_val % int(vocab_sizes[col].item())) + int(offsets[col].item())
    return {"ngram_token_ids": ngram_ids, "multipliers": multipliers,
            "vocab_sizes": vocab_sizes, "offsets": offsets}, {"output": output}


# ===========================================================================
# 31. DeepSeek: get_fused_mapping
# ===========================================================================

def verify_get_fused_mapping(NumTokens=16, NumTopk=8, NumExperts=32, Alignment=64):
    topk_idx = randint((NumTokens, NumTopk), low=0, high=NumExperts, dtype=torch.int32, seed=1)
    kNumel = NumTokens * NumTopk

    # Pass 1: histogram
    cnt = torch.bincount(topk_idx.flatten(), minlength=NumExperts)

    # Pass 2: aligned prefix sum
    expert_start = torch.zeros(NumExperts, dtype=torch.int32)
    expert_end = torch.zeros(NumExperts, dtype=torch.int32)
    num_tokens_per_expert = torch.zeros(NumExperts, dtype=torch.int32)
    acc = 0
    for e in range(NumExperts):
        aligned = ((cnt[e].item() + Alignment - 1) // Alignment) * Alignment
        expert_start[e] = acc
        expert_end[e] = acc + aligned
        num_tokens_per_expert[e] = aligned
        acc += aligned

    total_expanded = acc
    pos_to_expert = torch.full((total_expanded,), -1, dtype=torch.int32)
    pos_to_token = torch.full((total_expanded,), -1, dtype=torch.int32)
    pos_to_token_topk = torch.full((total_expanded,), -1, dtype=torch.int32)
    token_topk_to_pos = torch.full((NumTokens, NumTopk), -1, dtype=torch.int32)

    # Pass 3: fill expert ranges
    for e in range(NumExperts):
        for p in range(expert_start[e].item(), expert_end[e].item()):
            pos_to_expert[p] = e

    # Pass 4: place tokens
    cursor = expert_start.clone()
    for t in range(NumTokens):
        for k in range(NumTopk):
            e = topk_idx[t, k].item()
            if e >= 0 and e < NumExperts:
                p = cursor[e].item()
                if p < expert_end[e].item():
                    pos_to_token[p] = t
                    pos_to_token_topk[p] = k
                    token_topk_to_pos[t, k] = p
                    cursor[e] += 1

    return {"topk_idx": topk_idx}, {
        "pos_to_expert": pos_to_expert, "pos_to_token": pos_to_token,
        "pos_to_token_topk": pos_to_token_topk, "token_topk_to_pos": token_topk_to_pos,
        "expert_start": expert_start, "expert_end": expert_end,
        "num_tokens_per_expert": num_tokens_per_expert
    }


# ===========================================================================
# 32. DeepSeek: expand_to_fused
# ===========================================================================

def verify_expand_to_fused(NumTokens=16, Hidden=64, NumTopk=4, NumExpanded=32, dtype=torch.bfloat16):
    x = randn((NumTokens, Hidden), dtype=dtype, seed=1)
    token_topk_to_pos = randint((NumTokens, NumTopk), low=-1, high=NumExpanded, dtype=torch.int32, seed=2)
    NumExperts_local = 8; pos_to_expert = randint((NumExpanded,), low=-1, high=NumExperts_local, dtype=torch.int32, seed=3)

    expanded_x = torch.zeros(NumExpanded, Hidden, dtype=dtype)
    for n in range(NumTokens):
        for k in range(NumTopk):
            p = token_topk_to_pos[n, k].item()
            if p >= 0:
                expanded_x[p] = x[n]
    # zero invalid positions
    for p in range(NumExpanded):
        if pos_to_expert[p].item() < 0:
            expanded_x[p] = 0
    return {"x": x, "token_topk_to_pos": token_topk_to_pos,
            "pos_to_expert": pos_to_expert}, {"expanded_x": expanded_x}


# ===========================================================================
# Registry
# ===========================================================================

KERNELS = {
    "matmul":            verify_matmul,
    "fa_2d_unroll":      verify_fa_2d_unroll,
    "sfa":               verify_sfa,
    "fa_softmax":        verify_fa_softmax,
    "transpose_2d":      verify_transpose_2d,
    "transpose_nd":      verify_transpose_nd,
    "reducesum_col":     verify_reducesum_col,
    "reducesum_row":     verify_reducesum_row,
    "reducemax_col":     verify_reducemax_col,
    "reducemax_row":     verify_reducemax_row,
    "gelu":              verify_gelu,
    "broadcast":         verify_broadcast,
    "concat_gather":     verify_concat_gather,
    "gather":            verify_gather,
    "fused_weight":      verify_fused_weight,
    "rms_norm":          verify_rms_norm,
    "batched_transpose": verify_batched_transpose,
    "expand_mhc_fwd":    verify_expand_to_mhc_fwd,
    "expand_mhc_bwd":    verify_expand_to_mhc_bwd,
    "topk_gate":         verify_topk_gate,
    "normalize_weight":  verify_normalize_weight,
    "group_count":       verify_group_count,
    "aux_fi":            verify_aux_fi,
    "cast_back_token":   verify_cast_back_per_token,
    "cast_back_channel": verify_cast_back_per_channel,
    "per_token_cast":    verify_per_token_cast,
    "per_channel_cast":  verify_per_channel_cast,
    "swiglu":            verify_swiglu,
    "reduce_fused":      verify_reduce_fused,
    "inplace_unique":    verify_inplace_unique,
    "sinkhorn":          verify_sinkhorn,
    "fn_normw_merge":    verify_fn_normw_merge,
    "mask_indices_tp":   verify_mask_indices_by_tp,
    "engram_hash":       verify_engram_hash,
    "get_fused_mapping": verify_get_fused_mapping,
    "expand_to_fused":   verify_expand_to_fused,
}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="SuperNPUBench PyTorch verification")
    parser.add_argument("--kernel", type=str, default="all",
                        help="Kernel name (or 'all')")
    parser.add_argument("--print", action="store_true",
                        help="Print input/output tensor shapes")
    parser.add_argument("--export", type=str, default=None,
                        help="Export binary files to directory")
    args = parser.parse_args()

    names = list(KERNELS.keys()) if args.kernel == "all" else [args.kernel]

    for name in names:
        if name not in KERNELS:
            print(f"Unknown kernel: {name}")
            print(f"Available: {', '.join(KERNELS.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        inputs, outputs = KERNELS[name]()
        for key, val in inputs.items():
            if isinstance(val, torch.Tensor):
                print(f"  input  {key:30s} shape={tuple(val.shape)} dtype={val.dtype}")
                if args.export:
                    path = f"{args.export}/{name}_in_{key}.bin"
                    to_np(val).tofile(path)
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor):
                print(f"  output {key:30s} shape={tuple(val.shape)} dtype={val.dtype}")
                if args.export:
                    path = f"{args.export}/{name}_out_{key}.bin"
                    to_np(val).tofile(path)
        if args.print:
            for key, val in {**inputs, **outputs}.items():
                if isinstance(val, torch.Tensor) and val.numel() <= 32:
                    print(f"    {key} = {val.reshape(-1)}")

    print(f"\n{'='*60}")
    print(f"  Done. {len(names)} kernel(s) verified.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
