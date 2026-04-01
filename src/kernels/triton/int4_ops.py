from __future__ import annotations

import torch
import triton
import triton.language as tl


def _ceil_half(k: int) -> int:
    return (k + 1) // 2


def fp16_weight_bytes(n: int, k: int) -> int:
    return n * k * 2


def int4_packed_weight_bytes(n: int, k: int) -> int:
    return n * _ceil_half(k) + n * 4


def int4_vs_fp16_memory_ratio(n: int, k: int) -> float:
    return fp16_weight_bytes(n, k) / max(1, int4_packed_weight_bytes(n, k))


BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32
GEMM_NUM_WARPS = 4
GEMM_NUM_STAGES = 2


@triton.jit
def _quantize_pack_byte_kernel(
    w_ptr,
    packed_ptr,
    scale_ptr,
    n_rows,
    k_in,
    stride_w_n,
    stride_w_k,
    stride_p_n,
    stride_p_b,
):
    row = tl.program_id(0)
    b = tl.program_id(1)
    mask_row = row < n_rows
    mask_b = b * 2 < k_in
    active = mask_row & mask_b

    scale = tl.load(scale_ptr + row, mask_row, other=0.0)
    inv_scale = tl.where(scale > 0, 1.0 / scale, 0.0)

    k0 = b * 2
    k1 = k0 + 1

    off0 = row * stride_w_n + k0 * stride_w_k
    off1 = row * stride_w_n + k1 * stride_w_k
    w0 = tl.load(w_ptr + off0, mask=active, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + off1, mask=mask_row & (k1 < k_in), other=0.0).to(tl.float32)

    z0 = w0 * inv_scale
    z1 = w1 * inv_scale
    q0 = tl.extra.cuda.libdevice.rint(z0)
    q1 = tl.extra.cuda.libdevice.rint(z1)
    q0 = tl.maximum(-8.0, tl.minimum(7.0, q0))
    q1 = tl.maximum(-8.0, tl.minimum(7.0, q1))
    u0 = (q0.to(tl.int32) + 8) & 0xF
    u1 = (q1.to(tl.int32) + 8) & 0xF
    byte = tl.where(k1 < k_in, u0 | (u1 << 4), u0)

    p_off = row * stride_p_n + b * stride_p_b
    tl.store(packed_ptr + p_off, byte.to(tl.uint8), active)


def quantize_fp16_to_int4_packed(w_fp16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert w_fp16.dim() == 2
    assert w_fp16.is_cuda
    n, k = w_fp16.shape
    w = w_fp16.float()
    abs_max, _ = torch.max(torch.abs(w), dim=1)
    scales = torch.where(abs_max > 0, abs_max / 7.0, torch.ones_like(abs_max))

    k_bytes = _ceil_half(k)
    packed = torch.empty((n, k_bytes), device=w_fp16.device, dtype=torch.uint8)
    if n == 0 or k_bytes == 0:
        return packed, scales

    stride_w_n, stride_w_k = w_fp16.stride()
    if stride_w_k != 1:
        w_contig = w_fp16.contiguous()
        stride_w_n, stride_w_k = w_contig.stride()
        w_ptr = w_contig
    else:
        w_ptr = w_fp16

    grid = (n, k_bytes)
    _quantize_pack_byte_kernel[grid](
        w_ptr,
        packed,
        scales,
        n,
        k,
        stride_w_n,
        stride_w_k,
        packed.stride(0),
        packed.stride(1),
    )
    return packed, scales


@triton.jit
def _matmul_x_w4_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    out_ptr,
    m_size,
    n_size,
    k_in,
    stride_xm,
    stride_xk,
    stride_pn,
    stride_pb,
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

        byte_idx = k_idx[:, None] // 2
        p_off = offs_n[None, :] * stride_pn + byte_idx * stride_pb
        mask_w = (k_idx[:, None] < k_in) & mask_n[None, :]
        raw = tl.load(packed_ptr + p_off, mask=mask_w, other=0).to(tl.int32)
        nib_sel = (k_idx[:, None] % 2).to(tl.int32)
        shift = nib_sel * 4
        w_q = (raw >> shift) & 0xF
        scale_row = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
        w_bf16 = ((w_q.to(tl.float32) - 8.0) * scale_row[None, :]).to(tl.bfloat16)

        acc += tl.dot(x_blk, w_bf16)

    out_off = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr + out_off, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def matmul_x_bf16_w4(
    x_bf16: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    assert x_bf16.dim() == 2 and packed_w.dim() == 2 and scales.dim() == 1
    assert x_bf16.is_cuda and packed_w.is_cuda and scales.is_cuda
    m, k = x_bf16.shape
    n, kb = packed_w.shape
    assert scales.shape[0] == n
    assert _ceil_half(k) == kb

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)

    x = x_bf16.contiguous()
    p = packed_w.contiguous()
    s = scales.contiguous()

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))

    _matmul_x_w4_kernel[grid](
        x,
        p,
        s,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        p.stride(0),
        p.stride(1),
        out.stride(1),
        out.stride(0),
        num_warps=GEMM_NUM_WARPS,
        num_stages=GEMM_NUM_STAGES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out
