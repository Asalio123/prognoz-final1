"""Статистический стек — полная сетка origins (исправление 06.10 после rc0).

БАГ ПРЕЖНЕЙ ВЕРСИИ (rc0, P0-1): cross_validation(h=12, n_windows=12) на 24-точечных
рядах размещает окна с train-длинами 1..12; фильтр origin>=12 оставлял ровно одно окно
(origin 12). Итог: весь стат-стек, включая знаменатель RelMAE (SeasonalNaive), был
оценён на одном origin, а T1 смешивала несовместимые ячейки.

ПРАВИЛЬНАЯ СХЕМА (по замороженному configs/validation.yaml):
per-horizon cross_validation: h=1→12 окон (origins 12..23), h=3→10, h=6→7, h=12→1.
Оставляем ТОЛЬКО точный шаг step==h из каждого окна (точечный прогноз t+h).
Дополнительно сохраняем все шаги h=12-окна (origin 12, шаги 1..12) — профиль
деградации по шагу для отчёта (B3 §1.3).

Запуск: python scripts/02_run_stats.py <chunk_id> <n_chunks>
Чанки: N независимых процессов, n_jobs=1 каждый (n_jobs=-1 на macOS нестабилен).
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

# (горизонт, число окон): origins 12..23 для h=1, 12..21 для h=3, 12..17 для h=6, 12 для h=12
GRID = [(1, 12), (3, 10), (6, 7), (12, 1)]


def main(chunk_id: int, n_chunks: int, only: str = ""):
    models = [m for m in MODELS if not only or only.lower() in type(m).__name__.lower()]
    suffix = f"_{only.lower()}" if only else ""
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    df = panel.rename(columns={"date": "ds"})[["territory_id", "category", "ds", "log_value", "value"]]
    df["unique_id"] = df["territory_id"].astype(str) + "|" + df["category"]
    ids = np.array(sorted(df["unique_id"].unique()))
    my = set(ids[chunk_id::n_chunks].tolist())
    df = df[df["unique_id"].isin(my)]
    df["ds"] = pd.PeriodIndex(df["ds"], freq="M").to_timestamp()
    sf_in = df[["unique_id", "ds", "log_value"]].rename(columns={"log_value": "y"}).sort_values(["unique_id", "ds"])
    truth = df[["unique_id", "ds", "value"]]

    sf = StatsForecast(models=models, freq="MS", n_jobs=1,
                       fallback_model=SeasonalNaive(season_length=12))
    t0 = time.time()
    parts, steps12 = [], []
    for h, n_win in GRID:
        t1 = time.time()
        cv = sf.cross_validation(df=sf_in, h=h, step_size=1, n_windows=n_win, level=[])
        cv["origin"] = ((cv["cutoff"].dt.year - 2023) * 12 + cv["cutoff"].dt.month)
        cv["step"] = ((cv["ds"].dt.year - 2023) * 12 + cv["ds"].dt.month) - cv["origin"]
        n_before = len(cv)
        cv = cv[cv["origin"] >= 12]  # страховка; при правильной сетке фильтр не должен ничего резать
        dropped = n_before - len(cv)
        print(f"[chunk {chunk_id}] h={h}: {time.time()-t1:.0f}с, строк {len(cv)}"
              + (f" ВНИМАНИЕ отброшено {dropped}" if dropped else ""), flush=True)
        if h == 12:
            steps12.append(cv)  # все шаги для профиля деградации
        parts.append(cv[cv["step"] == h].assign(horizon=h))

    cv_all = pd.concat(parts, ignore_index=True).merge(truth, on=["unique_id", "ds"], how="left")
    model_cols = [c for c in cv_all.columns if c not in
                  ("unique_id", "ds", "cutoff", "origin", "step", "horizon", "value")]
    long = cv_all.melt(id_vars=["unique_id", "ds", "origin", "horizon", "value"],
                       value_vars=model_cols, var_name="model", value_name="y_pred_log")
    long["y_pred"] = np.expm1(long["y_pred_log"].clip(lower=0))
    long = long.rename(columns={"value": "y_true", "unique_id": "series_id"})
    out = ROOT / "outputs" / f"preds_stats{suffix}_part{chunk_id}.parquet"
    long[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(out)
    print(f"[chunk {chunk_id}] saved {out} ({len(long)} строк), {time.time()-t0:.0f}с total", flush=True)

    # профиль деградации: все шаги 1..12 единственного окна h=12 (origin 12)
    if steps12:
        s12 = pd.concat(steps12, ignore_index=True).merge(truth, on=["unique_id", "ds"], how="left")
        s12_long = s12.melt(id_vars=["unique_id", "ds", "origin", "step", "value"],
                            value_vars=model_cols, var_name="model", value_name="y_pred_log")
        s12_long["y_pred"] = np.expm1(s12_long["y_pred_log"].clip(lower=0))
        s12_long = s12_long.rename(columns={"value": "y_true", "step": "horizon", "unique_id": "series_id"})
        out12 = ROOT / "outputs" / f"preds_stats{suffix}_steps_part{chunk_id}.parquet"
        s12_long[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(out12)
        print(f"[chunk {chunk_id}] saved steps {out12}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
