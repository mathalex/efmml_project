from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernels.triton.int4_ops import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    GEMM_NUM_STAGES,
    GEMM_NUM_WARPS,
)


@triton.jit
def _matmul_x_w16_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    m_size,
    n_size,
    k_in,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_on,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < m_size
    mask_n = offs_n < n_size

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k_in, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)

        x_off = offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
        x_mask = mask_m[:, None] & (k_idx < k_in)[None, :]
        x_blk = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        w_off = offs_n[None, :] * stride_wn + k_idx[:, None] * stride_wk
        w_mask = mask_n[None, :] & (k_idx[:, None] < k_in)
        w_blk = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_blk, w_blk)

    out_off = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr + out_off, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def matmul_x_bf16_w16_triton(x_bf16: torch.Tensor, w_bf16: torch.Tensor) -> torch.Tensor:
    assert x_bf16.dim() == 2 and w_bf16.dim() == 2
    assert x_bf16.is_cuda and w_bf16.is_cuda
    m, k = x_bf16.shape
    n, kw = w_bf16.shape
    assert k == kw

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    x = x_bf16.contiguous()
    w = w_bf16.contiguous()

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))

    _matmul_x_w16_kernel[grid](
        x,
        w,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(1),
        out.stride(0),
        num_warps=GEMM_NUM_WARPS,
        num_stages=GEMM_NUM_STAGES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out
