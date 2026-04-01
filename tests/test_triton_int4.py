from __future__ import annotations

import pytest
import torch

from kernels.triton.int4_ops import (
    fp16_weight_bytes,
    int4_packed_weight_bytes,
    int4_vs_fp16_memory_ratio,
    matmul_x_bf16_w4,
    quantize_fp16_to_int4_packed,
)


def _unpack_int4(packed: torch.Tensor, k: int, scales: torch.Tensor) -> torch.Tensor:
    n, kb = packed.shape
    out = torch.zeros(n, k, device=packed.device, dtype=torch.float32)
    scales_f = scales.float().unsqueeze(1)
    for b in range(kb):
        byte = packed[:, b].to(torch.int32)
        lo = (byte & 0xF).float() - 8.0
        hi = ((byte >> 4) & 0xF).float() - 8.0
        out[:, 2 * b] = lo * scales_f[:, 0]
        if 2 * b + 1 < k:
            out[:, 2 * b + 1] = hi * scales_f[:, 0]
    return out


@pytest.fixture(scope="module")
def device():
    d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return d


def test_pack_memory_ratio(device):
    n, k = 128, 512
    w = torch.randn(n, k, device=device, dtype=torch.float16)
    packed, scales = quantize_fp16_to_int4_packed(w)
    assert packed.dtype == torch.uint8
    assert packed.shape == (n, (k + 1) // 2)
    assert scales.shape == (n,)
    assert int4_packed_weight_bytes(n, k) == n * packed.shape[1] + scales.numel() * 4
    assert int4_vs_fp16_memory_ratio(n, k) > 3.5
    assert int4_packed_weight_bytes(n, k) < fp16_weight_bytes(n, k) / 3.5


def test_matmul_close_to_torch_bf16(device):
    torch.manual_seed(0)
    cases = [(48, 64, 96), (32, 48, 127)]
    quantized = []
    for m, n, k in cases:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=device, dtype=torch.float16)
        packed, scales = quantize_fp16_to_int4_packed(w)
        quantized.append((m, n, k, x, packed, scales))

    for m, n, k, x, packed, scales in quantized:
        y_tri = matmul_x_bf16_w4(x, packed, scales)
        w_f = _unpack_int4(packed, k, scales)
        y_ref = torch.matmul(x, w_f.to(torch.bfloat16).t()).float()
        torch.testing.assert_close(y_tri.float(), y_ref, rtol=0.06, atol=0.12)
