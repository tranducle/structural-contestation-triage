from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core import fit_llr_table, score_llr, write_json
from lmd_shift_study import CORE_COLUMNS, load_and_engineer_lmd
from salad_detector_study import STRUCT_COLUMNS, USE_COLUMNS, engineer_salad


def _time_callable(function: Callable[[], Any], rows: int, warmup: int = 5, repeats: int = 30) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        durations.append(time.perf_counter() - started)
    array = np.asarray(durations, dtype=np.float64)
    median = float(np.median(array))
    return {
        "warmup_runs": int(warmup),
        "measured_runs": int(repeats),
        "rows": int(rows),
        "median_seconds": median,
        "interquartile_range_seconds": [float(np.quantile(array, 0.25)), float(np.quantile(array, 0.75))],
        "p95_seconds": float(np.quantile(array, 0.95)),
        "median_microseconds_per_row": float(median * 1e6 / max(1, rows)),
        "median_rows_per_second": float(rows / max(1e-12, median)),
        "all_seconds": durations,
    }


def benchmark_salad(paper_root: Path) -> dict[str, Any]:
    data_dir = paper_root / "data/external/salad_soc"
    sample_path = paper_root / "results/detector/train_sample.pkl"
    train = pd.read_pickle(sample_path)
    test = engineer_salad(pd.read_csv(data_dir / "salad_test.csv", usecols=USE_COLUMNS, low_memory=False))
    table = fit_llr_table(train, STRUCT_COLUMNS, "label", alpha=1.0)
    return _time_callable(lambda: score_llr(test, table, STRUCT_COLUMNS), len(test))


def benchmark_lmd(paper_root: Path) -> dict[str, Any]:
    path = paper_root / "data/external/lmd_2023/full-csv(Evaluated Titles).csv"
    frame = load_and_engineer_lmd(path)
    train = frame[frame["signature_bucket"] <= 6]
    test = frame[frame["signature_bucket"] >= 8]
    table = fit_llr_table(train, CORE_COLUMNS, "label", alpha=1.0)
    return _time_callable(lambda: score_llr(test, table, CORE_COLUMNS), len(test))


def run_runtime_benchmark(paper_root: Path) -> dict[str, Any]:
    result_dir = paper_root / "results/detector"
    payload = {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "thread_policy": "single Python process; pandas vectorized score mapping; no model training included",
        },
        "SALAD_PITSA_budget_matched": benchmark_salad(paper_root),
        "LMD_PITSA_signature_disjoint": benchmark_lmd(paper_root),
        "scope": "Inference scoring only. Data loading, feature engineering, fitting, and model training are excluded.",
    }
    rows = []
    for name in ["SALAD_PITSA_budget_matched", "LMD_PITSA_signature_disjoint"]:
        item = payload[name]
        rows.append({"benchmark": name, **{key: value for key, value in item.items() if key != "all_seconds"}})
    pd.DataFrame(rows).to_csv(result_dir / "runtime_results.csv", index=False)
    write_json(result_dir / "runtime_results.json", payload)
    return payload
