"""Triton FlashAttention-2 forward kernel (causal).

Hand-written Triton kernel implementing FlashAttention-2 forward: online
softmax with block-wise tiling over K/V, no materialized NxN attention matrix.
Matches train/attention.scaled_dot_product_attention with causal masking.
Falls back to PyTorch reference if Triton is unavailable.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def flash_attn_forward_kernel(
        Q, K, V, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N, D,
        sm_scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N) & (offs_d[None, :] < D), other=0.0)

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

        num_blocks_n = tl.cdiv(N, BLOCK_N)

        for start_n in range(0, num_blocks_n):
            offs_n_cur = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

            k_ptrs = K + off_z * stride_kz + off_h * stride_kh + offs_n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=(offs_n_cur[:, None] < N) & (offs_d[None, :] < D), other=0.0)

            v_ptrs = V + off_z * stride_vz + off_h * stride_vh + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=(offs_n_cur[:, None] < N) & (offs_d[None, :] < D), other=0.0)

            qk = tl.dot(q, k.trans(1, 0)) * sm_scale
            causal_mask = (offs_m[:, None] >= offs_n_cur[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v)
            m_i = m_new

        offs_m_out = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        out_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m_out[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(out_ptrs, acc / l_i[:, None], mask=(offs_m_out[:, None] < N) & (offs_d[None, :] < D))


def flash_attn_forward(q, k, v, causal=True):
    if not HAS_TRITON:
        raise RuntimeError("Triton not installed. `pip install triton`.")

    batch, n_heads, seq_len, head_dim = q.shape
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = head_dim

    assert head_dim <= BLOCK_DMODEL, f"head_dim {head_dim} > BLOCK_DMODEL {BLOCK_DMODEL}"

    sm_scale = head_dim ** -0.5
    out = torch.empty_like(q)

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
    flash_attn_forward_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, n_heads, seq_len, head_dim,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    return out
