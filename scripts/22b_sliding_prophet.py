"""Prophet-A / Prophet-B* на sliding-окне 18 мес — ТОЛЬКО срез «Все категории» (2016 рядов).

Бюджет (замеры 03: A=0.40 с/ряд, B=0.59 с/ряд wall на 8 jobs, 12 фитов/ряд):
полная панель 11682 рядов × 6 фитов ≈ 39-57 мин wall... но с запасом и nice — гранично;
срез 2016 × 6 фитов ≈ 7 (A) / 10 (B*) мин wall — принят. Полная панель Prophet
на sliding ОТКЛОНЕНА (>1ч с учётом nice и неопределённости), зафиксировано в отчёте.

Один фит на origin покрывает оба горизонта: origins 18..23 → h=1; origins 18..21 → h=3.
Train = месяцы [o-18, o) — ровно 18 точек (vs expanding [0, o)). Конфиги — как в
scripts/03_run_prophet.py (B* = yearly_seasonality=4, multiplicative, cps=0.05).

Запуск: nice -n 10 .venv/bin/python scripts/22b_sliding_prophet.py --version A --jobs 8
"""
import argparse
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["CMDSTAN_VERBOSE"] = "0"
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

WINDOW = 18
ORIGINS = list(range(18, 24))  # 18..23
MONTH0 = pd.Timestamp("2023-01-01")


def prophet_config(version: str) -> dict:
    if version == "A":
        return {}
    return dict(yearly_seasonality=4, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05)


def run_series(args):
    series_id, dates, y, version = args
    from prophet import Prophet
    cfg = prophet_config(version)
    out = []
    for o in ORIGINS:
        lo = o - WINDOW
        tr = pd.DataFrame({"ds": dates[lo:o], "y": y[lo:o]})
        hmax = min(3, 24 - o)  # один фит покрывает h=1 (все origins) и h=3 (origins<=21)
        if hmax < 1:
            continue
        try:
            m = Prophet(**cfg)
            m.fit(tr)
            fut = m.make_future_dataframe(periods=hmax, freq="MS")
            pr = m.predict(fut)["yhat"].values[-hmax:]
            for step in (1, 3):
                if step <= hmax:
                    out.append((series_id, o, step, max(float(pr[step - 1]), 0.0)))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["A", "B"], required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit-series", type=int, default=0)
    args = ap.parse_args()

    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel = panel[panel["category"] == "Все категории"]  # только headline-срез
    panel["ds"] = pd.PeriodIndex(panel["date"], freq="M").to_timestamp()
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]

    wide = panel.pivot_table(index="series_id", columns="ds", values="value", aggfunc="first")
    wide = wide.sort_index(axis=1)
    dates = list(wide.columns)
    series = [(sid, dates, wide.loc[sid].values.astype(float), args.version)
              for sid in wide.index]
    if args.limit_series:
        series = series[: args.limit_series]

    t0 = time.time()
    rows = []
    with get_context("spawn").Pool(args.jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(run_series, series, chunksize=8)):
            rows.extend(res)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{i+1}/{len(series)} рядов, {rate:.1f} ряд/с, "
                      f"ETA {(len(series)-i-1)/rate/60:.0f} мин", flush=True)
    print(f"Prophet-{args.version} sliding18: {len(series)} рядов за {time.time()-t0:.0f}с")

    pred = pd.DataFrame(rows, columns=["series_id", "origin", "horizon", "y_pred"])
    truth = panel[["series_id", "ds", "value"]].copy()
    truth["month_idx"] = (truth["ds"].dt.year - 2023) * 12 + truth["ds"].dt.month - 1
    pred["test_idx"] = pred["origin"] + pred["horizon"] - 1
    pred = pred.merge(truth[["series_id", "month_idx", "value"]],
                      left_on=["series_id", "test_idx"], right_on=["series_id", "month_idx"])
    pred = pred.rename(columns={"value": "y_true"})
    out = ROOT / "outputs" / f"robustness_sliding18_prophet{args.version}.parquet"
    pred[["series_id", "origin", "horizon", "y_true", "y_pred"]].assign(
        model=f"Prophet-{args.version}" + ("*" if args.version == "B" else "")).to_parquet(out)
    print("saved", out)


if __name__ == "__main__":
    main()
