from __future__ import annotations

import pytest
import torch

from kernels.triton.matmul_w16_triton import matmul_x_bf16_w16_triton


@pytest.fixture(scope="module")
def device():
    d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return d


def test_w16_triton_matches_torch(device):
    torch.manual_seed(1)
    for m, n, k in [(32, 48, 64), (128, 256, 512), (64, 2048, 2048)]:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        y_tri = matmul_x_bf16_w16_triton(x, w)
        y_ref = torch.matmul(x, w.t()).float()
        torch.testing.assert_close(y_tri.float(), y_ref, rtol=0.03, atol=0.08)
