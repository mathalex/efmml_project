#!/usr/bin/env python3
"""Собрать report/report.md из report/data/exp01_results.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parent
DATA_JSON = REPORT_DIR / "data" / "exp01_results.json"
OUT_MD = REPORT_DIR / "report.md"


def load_results():
    if not DATA_JSON.exists():
        return None
    return json.loads(DATA_JSON.read_text(encoding="utf-8"))


def ensure_ms_w16_triton(data: dict | None) -> dict | None:
    """При отсутствии ms_w16_triton в строках — замер на GPU и запись JSON (как в exp01)."""
    if not data or not data.get("rows"):
        return data
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from experiments.exp01_bench_matmul import fill_missing_ms_w16_triton_in_payload

    data, changed = fill_missing_ms_w16_triton_in_payload(data, None, None)
    if changed:
        DATA_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Updated {DATA_JSON} (measured ms_w16_triton on GPU).", file=sys.stderr)
    return data


def _normalize_bench_row(row: dict) -> dict:
    w16_torch = row.get("ms_w16_torch")
    if w16_torch is None:
        w16_torch = row.get("ms_w16")

    w16_triton = row.get("ms_w16_triton")
    w4 = row.get("ms_w4_triton")
    if w4 is None:
        w4 = row.get("ms_w4")

    ratio = row.get("w16_triton_over_w4")
    if ratio is None:
        ratio = row.get("speedup_w16_over_w4")
    if ratio is None and w16_triton is not None and w4 is not None:
        wt, wf = float(w16_triton), float(w4)
        if wt > 0:
            ratio = round(wf / wt, 3)

    return {
        "layer": row["layer"],
        "M": row["M"],
        "N": row["N"],
        "K": row["K"],
        "ms_w16_triton": float(w16_triton) if w16_triton is not None else None,
        "ms_w16_torch": float(w16_torch) if w16_torch is not None else None,
        "ms_w4_triton": float(w4) if w4 is not None else None,
        "w16_triton_over_w4": float(ratio) if ratio is not None else None,
    }


def _benchmark_md_table(data: dict) -> str:
    if not data or not data.get("rows"):
        return (
            f"*Таблица бенчмарка: сгенерируйте `{DATA_JSON.name}` командой*\n\n"
            f"`PYTHONPATH=src python experiments/exp01_bench_matmul.py "
            f"--output {DATA_JSON.as_posix()}`\n"
        )

    lines = [
        f"Параметры замера: `warmup={data.get('warmup')}`, `iters={data.get('iters')}` "
        f"(см. `bench()` в `experiments/exp01_bench_matmul.py`). "
        f"При сборке этого отчёта, если в JSON не было `ms_w16_triton`, значения дописываются "
        f"тем же ядром и прогревом (`measure_ms_w16_triton_grid` в том же файле).",
        "",
        "| layer | M | N | K | W16 Triton (ms) | W16 torch (ms) | W4 Triton (ms) | W4÷W16tr |",
        "|-------|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["rows"]:
        r = _normalize_bench_row(row)
        missing = [k for k in ("ms_w16_triton", "ms_w16_torch", "ms_w4_triton", "w16_triton_over_w4") if r[k] is None]
        if missing:
            raise ValueError(f"Incomplete bench row for {r['layer']} M={r['M']}: missing {missing}")
        lines.append(
            f"| {r['layer']} | {r['M']} | {r['N']} | {r['K']} | "
            f"{r['ms_w16_triton']:.4f} | {r['ms_w16_torch']:.4f} | {r['ms_w4_triton']:.4f} | "
            f"{r['w16_triton_over_w4']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown(data) -> None:
    table_and_bench = _benchmark_md_table(data or {})
    body = f"""# Квантизация весов LLM и инференс (Triton)

Отчёт по проекту efmml: эксперимент 1, затем план из `ORIGINAL_TASK.md` и расписание на 4 недели.

## 1. Эксперимент 1

Реализованы ядра в `src/kernels/triton/int4_ops.py` (квантизация + matmul W4) и
`matmul_w16_triton.py` (matmul W16 на Triton для сравнения). Цель по памяти: весы
в int4 занимают ~в 4 раза меньше, чем fp16 (см. `fp16_weight_bytes` / `int4_packed_weight_bytes`).

Корректность: `pytest tests/`.

### 1.1 Откуда берутся строки таблицы

Каждая **строка** — это один замер для пары *(слой, размер батча токенов M)*.

1. Список слоёв и размеров весов **(N, K)** задаётся в `src/llama_1b_shapes.py`: константа
   `LLAMA_1B_LINEAR_SHAPES` (имя проекции `layer` и выходная/входная размерность в терминах
   `nn.Linear`: вес `(N, K)`, прямой проход `y = x @ W.T`, то есть активация `(M, K)`).
2. Размеры **M** перебираются из `TOKEN_BATCH_SIZES` = `(128, 512, 2048)` — как в задании для матрицы активаций X.
3. В `experiments/exp01_bench_matmul.py` двойной цикл по слоям и по `M`: для каждой пары создаётся случайная
   `x` формы `(M, K)` в `bfloat16` на GPU, затем вызывается `bench()` (прогрев `warmup`, усреднение по `iters`,
   синхронизация CUDA).
4. Результаты пишутся в JSON **в порядке вложенных циклов** (все M для первого слоя, затем следующий слой).

Перед замером — **трёхфазный прогрев** по всем слоям и всем M (отдельно W16 Triton, отдельно `torch.matmul`,
отдельно W4 Triton), чтобы снизить риск сбоев JIT на Python 3.13.

### 1.2 Что означают колонки таблицы

| Колонка | Смысл |
|---------|--------|
| **layer** | Имя линейного слоя из `LLAMA_1B_LINEAR_SHAPES` (например `q_proj`, `gate_proj`). |
| **M** | Число строк матрицы активаций `(M, K)` (токены 128 / 512 / 2048). |
| **N** | `out_features` слоя (строка веса W). |
| **K** | `in_features` (вход для `x @ W.T`). |
| **W16 Triton (ms)** | Среднее время `matmul_x_bf16_w16_triton(x, W_bf16)`. |
| **W16 torch (ms)** | Среднее время `torch.matmul(x, W_bf16.t())` (обычно cuBLAS). |
| **W4 Triton (ms)** | Среднее время `matmul_x_bf16_w4(x, packed, scales)` после квантизации веса в int4. |
| **W4÷W16tr** | `(время W4) / (время W16 Triton)`; в JSON — `w16_triton_over_w4`. Если больше 1, W4 медленнее W16 Triton на этом замере. |

### 1.3 Результаты замеров

{table_and_bench}

## 2. План и расписание (по `ORIGINAL_TASK.md`, 4 недели)

Ниже пункты плана из задания связаны с неделями работы.

**Неделя 1.** Triton: квантизация fp16→int4 с упаковкой и экономия памяти ~×4 (п. 1); matmul `X_bf16 @ W_4^T` (п. 2); сравнение с W16 на формах Llama-3.2-1B-Instruct и M ∈ {{128, 512, 2048}} (п. 3); тесты `pytest`; бенчмарк `exp01_bench_matmul.py`; черновик отчёта (`report/build.py`).

**Неделя 2.** Доработка ядер при необходимости; подготовка к квантизованному `nn.Linear` (п. 4).

**Неделя 3.** Квантизованный линейный слой на базе кернелей; загрузка весов Llama-3.2-1B-Instruct (п. 4).

**Неделя 4.** Замена линейных слоёв в модели; замеры скорости и перплексия на wikitext2 (п. 5); финальный отчёт.

Дополнительно по проекту: при необходимости CUDA-ядра и `torch.compile`.
"""
    OUT_MD.write_text(body, encoding="utf-8")
    print(f"Wrote {OUT_MD}")


def main() -> None:
    data = load_results()
    data = ensure_ms_w16_triton(data)
    write_markdown(data)


if __name__ == "__main__":
    main()
