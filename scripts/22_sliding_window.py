"""Робастность: sliding-window (18 мес) вместо expanding — проверка B3 §1.3 (robustness).

Схема: для origin t train = месяцы [t-18, t) (НЕ [0, t)), тест = t+h-1.
  h=1: origins 18..23 (6 окон), h=3: origins 18..21 (4 окна).
Дешёвые модели на полной панели (11682 ряда): SeasonalNaive, DynamicOptimizedTheta,
AutoETS (StatsForecast cross_validation с input_size=18) + DriftAdjSNaive (векторно,
та же формула, что scripts/08_run_seasadj.py — pooled YoY дрейф на origin).

Замечание по построению: SeasonalNaive и DriftAdjSNaive смотрят только на месяцы
[t-13, t) → их sliding-прогнозы ТОЖДЕСТВЕННЫ expanding на origins>=18; проверка
информативна для оконно-чувствительных моделей (DOT, AutoETS, Prophet).

Prophet дорогой: полная панель = 11682 рядов × 6 фитов × ~0.4-0.6 с/ряд ≈ 65-96 мин wall
(замер 03: A=0.40 с/ряд, B=0.59 с/ряд на 8 jobs) → >1ч, ОТКЛОНЁН. Prophet-A и Prophet-B*
считаются только на срезе «Все категории» (2016 рядов, ~7/10 мин wall) — scripts/22b.

Запуск: nice -n 10 .venv/bin/python scripts/22_sliding_window.py <chunk_id> <n_chunks>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, DynamicOptimizedTheta, SeasonalNaive

MODELS = [SeasonalNaive(season_length=12), DynamicOptimizedTheta(), AutoETS()]
GRID = [(1, 6), (3, 4)]          # (h, n_windows) → origins 18..23 и 18..21
WINDOW = 18
ORIGINS_H1 = list(range(18, 24))  # для DriftAdjSNaive
ORIGINS_H3 = list(range(18, 22))


def drift_adj_preds(panel_full: pd.DataFrame, keep_ids: set) -> pd.DataFrame:
    """DriftAdjSNaive на sliding-сетке. Формула идентична 08_run_seasadj.py:
    прогноз = y[t+h-1-12] + g, g = pooled медиана YoY (месяц t-1 против t-13).
    g считается по ПОЛНОЙ панели (как у чемпиона), не по чанку — иначе сдвиг константы."""
    panel_full = panel_full.copy()
    panel_full["month_idx"] = (pd.PeriodIndex(panel_full["date"], freq="M").year - 2023) * 12 + \
                              pd.PeriodIndex(panel_full["date"], freq="M").month - 1
    wide = panel_full.pivot_table(index="series_id", columns="month_idx", values="log_value").reindex(columns=range(24))
    wide_rub = panel_full.pivot_table(index="series_id", columns="month_idx", values="value").reindex(columns=range(24))
    wv = wide.to_numpy()
    truth = wide_rub.stack().rename("y_true").reset_index().rename(columns={"month_idx": "test_idx"})
    keep = wide.index.isin(list(keep_ids))
    wide_idx = wide.index[keep]
    wv = wv[keep]
    preds = []
    for o, h in [(o, h) for o in ORIGINS_H1 for h in (1, 3) if o in (ORIGINS_H1 if h == 1 else ORIGINS_H3)]:
        c = o - 1
        g = float(np.nanmedian(wide.to_numpy()[:, c] - wide.to_numpy()[:, c - 12]))  # pooled, c >= 17
        tm = c + h
        p_log = wv[:, tm - 12] + g
        preds.append(pd.DataFrame({"series_id": wide_idx, "origin": o, "horizon": h,
                                   "y_pred": np.expm1(np.clip(p_log, 0, None)),
                                   "model": "DriftAdjSNaive", "test_idx": tm}))
    pred = pd.concat(preds, ignore_index=True).merge(truth, on=["series_id", "test_idx"], how="left")
    return pred[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]]


def main(chunk_id: int, n_chunks: int):
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    ids = np.array(sorted(panel["series_id"].unique()))
    my = set(ids[chunk_id::n_chunks].tolist())
    df = panel[panel["series_id"].isin(my)].rename(columns={"date": "ds"})
    df["ds"] = pd.PeriodIndex(df["ds"], freq="M").to_timestamp()
    sf_in = df[["series_id", "ds", "log_value"]].rename(
        columns={"series_id": "unique_id", "log_value": "y"}).sort_values(["unique_id", "ds"])
    truth = df[["series_id", "ds", "value"]]

    sf = StatsForecast(models=MODELS, freq="MS", n_jobs=1,
                       fallback_model=SeasonalNaive(season_length=12))
    t0 = time.time()
    parts = []
    for h, n_win in GRID:
        t1 = time.time()
        cv = sf.cross_validation(df=sf_in, h=h, step_size=1, n_windows=n_win,
                                 input_size=WINDOW, level=[])
        cv["origin"] = ((cv["cutoff"].dt.year - 2023) * 12 + cv["cutoff"].dt.month)
        cv["step"] = ((cv["ds"].dt.year - 2023) * 12 + cv["ds"].dt.month) - cv["origin"]
        assert cv["origin"].min() >= 18, f"origin < 18 просочился: {cv['origin'].min()}"
        print(f"[chunk {chunk_id}] h={h}: {time.time()-t1:.0f}с, строк {len(cv)}, "
              f"origins {sorted(cv.origin.unique())}", flush=True)
        parts.append(cv[cv["step"] == h].assign(horizon=h))

    cv_all = pd.concat(parts, ignore_index=True).merge(truth, left_on=["unique_id", "ds"],
                                                       right_on=["series_id", "ds"], how="left")
    model_cols = [c for c in cv_all.columns if c not in
                  ("unique_id", "series_id", "ds", "cutoff", "origin", "step", "horizon", "value", "y")]
    long = cv_all.melt(id_vars=["series_id", "ds", "origin", "horizon", "value"],
                       value_vars=model_cols, var_name="model", value_name="y_pred_log")
    long["y_pred"] = np.expm1(long["y_pred_log"].clip(lower=0))
    long = long.rename(columns={"value": "y_true"})
    out = long[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]]

    # DriftAdjSNaive — векторно; g pooled по полной панели, прогнозы для рядов чанка
    da = drift_adj_preds(panel, my)
    out = pd.concat([out, da], ignore_index=True)

    fout = ROOT / "outputs" / f"robustness_sliding18_part{chunk_id}.parquet"
    out.to_parquet(fout, index=False)
    print(f"[chunk {chunk_id}] saved {fout} ({len(out)} строк), {time.time()-t0:.0f}с total", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
