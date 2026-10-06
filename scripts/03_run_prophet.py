"""Prophet A/B по замороженной сетке (configs/champion.yaml).

A = default (на спане 700 дней молча без yearly-сезонности — порог 730 в исходнике Prophet).
B = честный: yearly_seasonality=4, multiplicative, changepoint_prior_scale из сетки.
Победа засчитывается только против B. Prophet медленный (~0.5–2с/серия) → масштабируемость
через --sample (стратифицированная подвыборка МО) и multiprocessing.
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

HORIZONS = [1, 3, 6, 12]
ORIGINS = list(range(12, 24))
MONTH0 = pd.Timestamp("2023-01-01")


def prophet_config(version: str) -> dict:
    if version == "A":
        return {}  # sklearn-default; yearly_seasonality молча off при спане <730 дней
    # B: yearly_seasonality=4 + multiplicative. changepoint_prior_scale=0.05 = дефолт
    # Prophet (сетка из B1 фактически не прогонялась — зафиксировано аудитом rc0;
    # см. отчёт §К2). rc0 показал: при origin=12 (train 334 дня < 365.25) yearly=4
    # недоопределена и даёт катастрофу; честный B = splice origin 12 → A
    # (scripts/03b_prophet_b_fixed.py).
    return dict(yearly_seasonality=4, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05)


def run_series(args):
    series_id, dates, y, origins, version = args
    from prophet import Prophet
    cfg = prophet_config(version)
    out = []
    for o in origins:
        train_end = MONTH0 + pd.DateOffset(months=o)
        hmax = min(12, 24 - o)
        if hmax < 1:
            continue
        tr = pd.DataFrame({"ds": dates[:o], "y": y[:o]})
        try:
            m = Prophet(**cfg)
            m.fit(tr)
            fut = m.make_future_dataframe(periods=hmax, freq="MS")
            pr = m.predict(fut)["yhat"].values[-hmax:]
            for step in HORIZONS:
                if step <= hmax:
                    out.append((series_id, o, step, max(float(pr[step - 1]), 0.0)))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["A", "B"], required=True)
    ap.add_argument("--sample", type=int, default=0, help="N случайных МО (0 = вся панель)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit-series", type=int, default=0, help="для смоук-теста: первые N рядов")
    args = ap.parse_args()

    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["ds"] = pd.PeriodIndex(panel["date"], freq="M").to_timestamp()
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]

    if args.sample:
        rng = np.random.default_rng(42)
        mos = rng.choice(panel["territory_id"].unique(), size=args.sample, replace=False)
        panel = panel[panel["territory_id"].isin(set(mos.tolist()))]

    wide = panel.pivot_table(index="series_id", columns="ds", values="value", aggfunc="first")
    wide = wide.sort_index(axis=1)
    dates = list(wide.columns)
    series = [(sid, dates, wide.loc[sid].values.astype(float), ORIGINS, args.version)
              for sid in wide.index]
    if args.limit_series:
        series = series[: args.limit_series]

    t0 = time.time()
    rows = []
    with get_context('spawn').Pool(args.jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(run_series, series, chunksize=4)):
            rows.extend(res)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(series) - i - 1) / rate / 60
                print(f"{i+1}/{len(series)} рядов, {rate:.1f} ряд/с, ETA {eta:.0f} мин", flush=True)
    dt_s = time.time() - t0
    print(f"Prophet-{args.version}: {len(series)} рядов за {dt_s:.0f}с ({dt_s/len(series):.2f} с/ряд)")

    pred = pd.DataFrame(rows, columns=["series_id", "origin", "horizon", "y_pred"])
    truth = panel[["series_id", "ds", "value"]].copy()
    truth["month_idx"] = (truth["ds"].dt.year - 2023) * 12 + truth["ds"].dt.month - 1
    pred["test_idx"] = pred["origin"] + pred["horizon"] - 1
    pred = pred.merge(truth[["series_id", "month_idx", "value"]],
                      left_on=["series_id", "test_idx"], right_on=["series_id", "month_idx"])
    pred = pred.rename(columns={"value": "y_true"})
    out = ROOT / "outputs" / f"preds_prophet_{args.version}{'_s'+str(args.sample) if args.sample else ''}.parquet"
    pred[["series_id", "origin", "horizon", "y_true", "y_pred"]].assign(model=f"Prophet-{args.version}").to_parquet(out)
    print("saved", out)


if __name__ == "__main__":
    main()
