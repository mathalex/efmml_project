**Дисклеймер**. Тексты писались совместно с ИИ (например, вот этот README ниже). Код самих кернелов писался самостоятельно с чтением документации. Бенчмаркинг напиcал сначала сам, но до подробного состояния довёл через ИИ. "Оприличивание" проекта (make и прочее) тоже помог сделать ИИ (т.к. это всё рутинная работа -- без помощи ИИ бы тоже всё работало, но выглядело бы более "лабораторно", не так красиво). Отчёт находится в папке `report` в формате pdf.

# Проект по EfMML

Квантизация весов LLM (int4) и инференс: Triton-ядра, дальше — CUDA, слой в PyTorch, перплексия.

## Структура

| Путь | Назначение |
|------|------------|
| `ORIGINAL_TASK.md` | формулировка задания |
| `src/kernels/triton/int4_ops.py` | квантизация fp16→int4 (uint8) + matmul `X_bf16 @ W_4^T` |
| `src/kernels/triton/matmul_w16_triton.py` | matmul `X_bf16 @ W_bf16^T` на Triton (сравнение с W4) |
| `src/kernels/triton/examples/` | примеры стиля Triton |
| `src/llama_1b_shapes.py` | типичные размеры линейных слоёв Llama-3.2-1B-Instruct |
| `tests/` | pytest |
| `experiments/exp01_bench_matmul.py` | эксп. 1: время W4 vs W16 (Triton и torch) |
| `report/` | `report.md`, `data/exp01_results.json` |
| `report/build.py` | генерация `report.md` |

## Окружение (Python)

```bash
python3.13 -m venv .venv
.venv/bin/pip install torch triton --index-url https://download.pytorch.org/whl/cu124
# dev (pytest, numpy): из корня репозитория
.venv/bin/pip install -e ".[dev]"
```

Настройки pytest (`pythonpath = ["src"]`) заданы в **`pyproject.toml`**.

## Команды

```bash
.venv/bin/pytest tests/ -v
PYTHONPATH=src .venv/bin/python experiments/exp01_bench_matmul.py --output report/data/exp01_results.json
.venv/bin/python report/build.py
```

## Отчёт (Markdown)

`report/build.py` создаёт **`report/report.md`**. Если в `report/data/exp01_results.json` нет поля `ms_w16_triton` (старый формат), сборка дописывает его **замером на GPU** (`measure_ms_w16_triton_grid` в `experiments/exp01_bench_matmul.py`) и обновляет JSON; без CUDA в этом случае сначала запустите полный бенчмарк с `--output`.

## Память и matmul

- **Память весов:** int4 + scale на строку — около **×4** меньше, чем fp16 (`fp16_weight_bytes` / `int4_packed_weight_bytes`).
- **Matmul:** фиксированные плитки 32×32×32 в `int4_ops.py`. Сравнение скорости — в бенчмарке.

`exp01_bench_matmul.py` делает **трёхфазный прогрев** (все W16 Triton → все `torch.matmul` → все W4), иначе на Python 3.13 возможен сбой при подгрузке JIT.

## Замечание (Python 3.13 + Triton)

Повторная JIT-компиляция квантизации с **новым** `(N, K)` после matmul в том же процессе может ронять интерпретатор. В тестах сначала все `quantize_*`, затем matmul; в бенчмарке — сначала квантизация всех слоёв.

## Один файл зависимостей

Используется только **`pyproject.toml`**: метаданные проекта, `[project.optional-dependencies] dev` для pytest/numpy, секция `[tool.pytest.ini_options]`. Отдельный `requirements.txt` не нужен: ставьте `pip install -e ".[dev]"` (и torch/triton отдельно с индексом PyTorch).
