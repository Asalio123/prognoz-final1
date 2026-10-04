"""Статистический стек, чанк-версия: N независимых процессов, каждый n_jobs=1.

Причина: StatsForecast(n_jobs=-1) повисает на macOS под нагрузкой (Pool+numba).
Каждый чанк пишет outputs/preds_stats_part{i}.parquet; 04_build_tables.py их склеивает.
Запуск: python scripts/02_run_stats.py <chunk_id> <n_chunks>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from statsforecast import StatsForecast
from statsforecast.models import (AutoARIMA, AutoCES, AutoETS, AutoTheta,
                                  DynamicOptimizedTheta, HistoricAverage,
                                  Naive, SeasonalNaive)

MODELS = [
    Naive(),
    HistoricAverage(),
    SeasonalNaive(season_length=12),
    AutoETS(),
    AutoTheta(),
    AutoCES(),
    DynamicOptimizedTheta(),
    AutoARIMA(),
]


def main(chunk_id: int, n_chunks: int):
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    df = panel.rename(columns={"date": "ds"})[["territory_id", "category", "ds", "log_value", "value"]]
    df["unique_id"] = df["territory_id"].astype(str) + "|" + df["category"]
    ids = np.array(sorted(df["unique_id"].unique()))
    my = set(ids[chunk_id::n_chunks].tolist())
    df = df[df["unique_id"].isin(my)]
    df["ds"] = pd.PeriodIndex(df["ds"], freq="M").to_timestamp()
    sf_in = df[["unique_id", "ds", "log_value"]].rename(columns={"log_value": "y"}).sort_values(["unique_id", "ds"])

    sf = StatsForecast(models=MODELS, freq="MS", n_jobs=1,
                       fallback_model=SeasonalNaive(season_length=12))
    t0 = time.time()
    cv = sf.cross_validation(df=sf_in, h=12, step_size=1, n_windows=12, level=[])
    print(f"[chunk {chunk_id}] cv: {time.time()-t0:.0f}с, строк {len(cv)}", flush=True)

    cv["origin"] = ((cv["cutoff"].dt.year - 2023) * 12 + cv["cutoff"].dt.month)  # 12..23
    cv["step"] = ((cv["ds"].dt.year - 2023) * 12 + cv["ds"].dt.month) - cv["origin"]
    cv = cv[cv["origin"] >= 12]

    truth = df[["unique_id", "ds", "value"]]
    cv = cv.merge(truth, on=["unique_id", "ds"], how="left")

    model_cols = [c for c in cv.columns if c not in
                  ("unique_id", "ds", "cutoff", "origin", "step", "value")]
    long = cv.melt(id_vars=["unique_id", "ds", "origin", "step", "value"],
                   value_vars=model_cols, var_name="model", value_name="y_pred_log")
    long = long[long["step"].isin([1, 3, 6, 12])]
    long["y_pred"] = np.expm1(long["y_pred_log"].clip(lower=0))
    long = long.rename(columns={"value": "y_true", "step": "horizon", "unique_id": "series_id"})
    out = ROOT / "outputs" / f"preds_stats_part{chunk_id}.parquet"
    long[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(out)
    print(f"[chunk {chunk_id}] saved {out}, {time.time()-t0:.0f}с total", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
