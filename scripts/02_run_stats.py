"""Полный прогон статистического стека по замороженной схеме валидации.

StatsForecast.cross_validation(h=12, step_size=1, n_windows=12) = expanding rolling-origin
origins 12..23 (ровно configs/validation.yaml). Таргет log1p, оценка после expm1 в рублях.
Выход: outputs/preds_stats.parquet (series_id, origin, horizon, y_true, y_pred, model).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from statsforecast import StatsForecast
from statsforecast.models import (AutoARIMA, AutoCES, AutoETS, AutoTheta,
                                  DynamicOptimizedTheta, HistoricAverage,
                                  Naive, SeasonalNaive)

MODELS = [
    Naive(),                       # пол: последнее значение
    HistoricAverage(),             # пол: среднее
    SeasonalNaive(season_length=12),  # скамья для RelMAE
    AutoETS(),
    AutoTheta(),
    AutoCES(),
    DynamicOptimizedTheta(),
    AutoARIMA(),                   # эталон Nixtla-бенчмарков (B1)
]


def main():
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    df = panel.rename(columns={"date": "ds"})[["territory_id", "category", "ds", "log_value", "value"]]
    df["unique_id"] = df["territory_id"].astype(str) + "|" + df["category"]
    df["ds"] = pd.PeriodIndex(df["ds"], freq="M").to_timestamp()
    sf_in = df[["unique_id", "ds", "log_value"]].rename(columns={"log_value": "y"}).sort_values(["unique_id", "ds"])

    sf = StatsForecast(models=MODELS, freq="MS", n_jobs=-1, fallback_model=SeasonalNaive(season_length=12))
    t0 = time.time()
    cv = sf.cross_validation(df=sf_in, h=12, step_size=1, n_windows=12, level=[])
    print(f"cross_validation: {time.time()-t0:.0f}с, строк {len(cv)}")

    # cutoff = последний месяц train (0-индекс месяца → origin = месяц после cutoff)
    month0 = pd.Timestamp("2023-01-01")
    cv["origin"] = ((cv["cutoff"].dt.year - 2023) * 12 + cv["cutoff"].dt.month)  # 1..23
    cv["step"] = ((cv["ds"].dt.year - 2023) * 12 + cv["ds"].dt.month) - cv["origin"] + 1  # 1..12
    cv = cv[cv["origin"] >= 12]

    # факт в рублях
    truth = df[["unique_id", "ds", "value"]]
    cv = cv.merge(truth, on=["unique_id", "ds"], how="left")

    model_cols = [c for c in cv.columns if c not in
                  ("unique_id", "ds", "cutoff", "origin", "step", "value")]
    long = cv.melt(id_vars=["unique_id", "ds", "origin", "step", "value"],
                   value_vars=model_cols, var_name="model", value_name="y_pred_log")
    long = long[long["step"].isin([1, 3, 6, 12])]  # только замороженные горизонты
    long["y_pred"] = np.expm1(long["y_pred_log"].clip(lower=0))
    long = long.rename(columns={"value": "y_true", "step": "horizon", "unique_id": "series_id"})
    long[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(
        ROOT / "outputs" / "preds_stats.parquet")
    print("saved outputs/preds_stats.parquet")


if __name__ == "__main__":
    main()
