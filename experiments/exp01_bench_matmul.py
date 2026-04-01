#!/usr/bin/env python3
"""Experiment 1: benchmark X@W4^T (Triton) vs X@W16^T (Triton и torch) на формах Llama-3.2-1B."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from kernels.triton.int4_ops import (
    fp16_weight_bytes,
    int4_packed_weight_bytes,
    int4_vs_fp16_memory_ratio,
    matmul_x_bf16_w4,
    quantize_fp16_to_int4_packed,
)
from kernels.triton.matmul_w16_triton import matmul_x_bf16_w16_triton
from llama_1b_shapes import LLAMA_1B_LINEAR_SHAPES, TOKEN_BATCH_SIZES


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters


def build_layers_data(device: torch.device):
    """Слой → (layer_meta, packed, scales, w_bf16). Квантизация всех весов до matmul."""
    layers_data: list[tuple] = []
    for layer in LLAMA_1B_LINEAR_SHAPES:
        n, k = layer.n, layer.k
        w_fp16 = torch.randn(n, k, device=device, dtype=torch.float16)
        packed, scales = quantize_fp16_to_int4_packed(w_fp16)
        w_bf16 = w_fp16.to(torch.bfloat16)
        layers_data.append((layer, packed, scales, w_bf16))
    return layers_data


def phased_warmup(layers_data: list[tuple]) -> None:
    """Трёхфазный прогрев (Python 3.13 + Triton): W16 Triton → torch → W4."""
    for layer, packed, scales, w_bf16 in layers_data:
        n, k = layer.n, layer.k
        for m in TOKEN_BATCH_SIZES:
            x = torch.randn(m, k, device=w_bf16.device, dtype=torch.bfloat16)
            matmul_x_bf16_w16_triton(x, w_bf16)
    _sync()
    for layer, packed, scales, w_bf16 in layers_data:
        n, k = layer.n, layer.k
        for m in TOKEN_BATCH_SIZES:
            x = torch.randn(m, k, device=w_bf16.device, dtype=torch.bfloat16)
            torch.matmul(x, w_bf16.t())
    _sync()
    for layer, packed, scales, w_bf16 in layers_data:
        n, k = layer.n, layer.k
        for m in TOKEN_BATCH_SIZES:
            x = torch.randn(m, k, device=w_bf16.device, dtype=torch.bfloat16)
            matmul_x_bf16_w4(x, packed, scales)
    _sync()


def measure_ms_w16_triton_grid(
    warmup: int, iters: int, device: torch.device | None = None
) -> dict[tuple[str, int], float]:
    """Замер только W16 Triton (ms) для каждой пары (layer.name, M); тот же прогрев, что в полном бенчмарке."""
    if device is None:
        device = torch.device("cuda:0")
    layers_data = build_layers_data(device)
    phased_warmup(layers_data)
    out: dict[tuple[str, int], float] = {}
    for layer, packed, scales, w_bf16 in layers_data:
        n, k = layer.n, layer.k
        for m in TOKEN_BATCH_SIZES:
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)

            def run_w16_triton():
                return matmul_x_bf16_w16_triton(x, w_bf16)

            t = bench(run_w16_triton, warmup, iters)
            out[(layer.name, m)] = round(t * 1000, 4)
    return out


def fill_missing_ms_w16_triton_in_payload(
    data: dict, warmup: int | None, iters: int | None
) -> tuple[dict, bool]:
    """Дописать ms_w16_triton и пересчитать отношения, если в JSON их нет (нужна CUDA). Возвращает (data, изменяли ли JSON)."""
    if not data.get("rows"):
        return data, False
    w = warmup if warmup is not None else int(data.get("warmup", 5))
    i = iters if iters is not None else int(data.get("iters", 20))
    need = any(r.get("ms_w16_triton") is None for r in data["rows"])
    if not need:
        return data, False
    if not torch.cuda.is_available():
        raise RuntimeError(
            "В JSON нет ms_w16_triton; для пересчёта нужна CUDA. "
            "Запустите: PYTHONPATH=src python experiments/exp01_bench_matmul.py "
            "--output report/data/exp01_results.json"
        )
    grid = measure_ms_w16_triton_grid(w, i)
    for r in data["rows"]:
        if r.get("ms_w16_triton") is not None:
            continue
        key = (r["layer"], int(r["M"]))
        if key not in grid:
            raise KeyError(f"Нет замера для {key}")
        r["ms_w16_triton"] = grid[key]
        w4 = r.get("ms_w4_triton", r.get("ms_w4"))
        w16t = float(r["ms_w16_triton"])
        w16cu = r.get("ms_w16_torch", r.get("ms_w16"))
        if w4 is not None and w16t > 0:
            wf = float(w4)
            r["w16_triton_over_w4"] = round(wf / w16t, 3)
            r["w4_over_w16_triton"] = round(w16t / wf, 3)
        if w16cu is not None and float(w16cu) > 0:
            r["w16_torch_over_w16_triton"] = round(w16t / float(w16cu), 3)
    return data, True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda:0")
    results: list[dict] = []

    print("Память весов W (только матрица, без активаций):")
    print(f"{'layer':<12} {'N':>6} {'K':>6} {'fp16 MiB':>10} {'int4 MiB':>10} {'× экономия':>12}")
    print("-" * 52)
    for layer in LLAMA_1B_LINEAR_SHAPES:
        nb, kb = layer.n, layer.k
        fp_b = fp16_weight_bytes(nb, kb)
        q_b = int4_packed_weight_bytes(nb, kb)
        ratio = int4_vs_fp16_memory_ratio(nb, kb)
        print(
            f"{layer.name:<12} {nb:>6} {kb:>6} "
            f"{fp_b / 2**20:>10.3f} {q_b / 2**20:>10.3f} {ratio:>11.2f}×"
        )
    print()

    layers_data = build_layers_data(device)
    phased_warmup(layers_data)

    for layer, packed, scales, w_bf16 in layers_data:
        n, k = layer.n, layer.k
        for m in TOKEN_BATCH_SIZES:
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)

            def run_w16_triton():
                return matmul_x_bf16_w16_triton(x, w_bf16)

            def run_w16_torch():
                return torch.matmul(x, w_bf16.t())

            def run_w4():
                return matmul_x_bf16_w4(x, packed, scales)

            t_w16_tri = bench(run_w16_triton, args.warmup, args.iters)
            t_w16_cu = bench(run_w16_torch, args.warmup, args.iters)
            t_w4 = bench(run_w4, args.warmup, args.iters)

            results.append(
                {
                    "layer": layer.name,
                    "N": n,
                    "K": k,
                    "M": m,
                    "ms_w16_triton": round(t_w16_tri * 1000, 4),
                    "ms_w16_torch": round(t_w16_cu * 1000, 4),
                    "ms_w4_triton": round(t_w4 * 1000, 4),
                    "w4_over_w16_triton": round(t_w16_tri / t_w4, 3),
                    "w16_triton_over_w4": round(t_w4 / t_w16_tri, 3),
                    "w16_torch_over_w16_triton": round(t_w16_tri / t_w16_cu, 3),
                }
            )

    text_lines = [
        f"{'layer':<12} {'M':>6} {'N':>6} {'K':>6} "
        f"{'W16 trit':>9} {'W16 torch':>10} {'W4 trit':>9} {'W4/W16tr':>8}",
        "-" * 70,
    ]
    for r in results:
        text_lines.append(
            f"{r['layer']:<12} {r['M']:>6} {r['N']:>6} {r['K']:>6} "
            f"{r['ms_w16_triton']:>9.4f} {r['ms_w16_torch']:>10.4f} {r['ms_w4_triton']:>9.4f} "
            f"{r['w16_triton_over_w4']:>8.3f}"
        )
    report = "\n".join(text_lines)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"warmup": args.warmup, "iters": args.iters, "rows": results}
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
