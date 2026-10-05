"""Chronos-2 zero-shot по замороженной сетке (К4). Один батч-вызов на origin с
prediction_length = 24 - origin, срез горизонтов 1/3/6/12 (R06: 12 прогонов вместо 30).

Таргет log1p, оценка в рублях после expm1 (медиана квантильного прогноза).
chronos-forecasting>=2.3.2 обязателен: наши уровни (14–91k руб) в диапазоне bf16-бага.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6, 12]
ORIGINS = list(range(12, 24))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--tag", default="chronos2_zs_log")
    args = ap.parse_args()

    from chronos import Chronos2Pipeline
    t0 = time.time()
    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map=args.device)
    print(f"load {time.time()-t0:.0f}с, device={args.device}", flush=True)

    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    panel["month_idx"] = (pd.PeriodIndex(panel["date"], freq="M").year - 2023) * 12 + \
                         pd.PeriodIndex(panel["date"], freq="M").month - 1
    wide = panel.pivot_table(index="series_id", columns="month_idx", values="log_value").reindex(columns=range(24))
    wide_rub = panel.pivot_table(index="series_id", columns="month_idx", values="value").reindex(columns=range(24))
    if args.limit:
        wide, wide_rub = wide.iloc[: args.limit], wide_rub.iloc[: args.limit]
    wv = wide.to_numpy()
    sids = wide.index.to_numpy()

    preds = []
    for o in ORIGINS:
        hmax = 24 - o
        ctx = [wv[i, :o] for i in range(len(wv))]
        t1 = time.time()
        fc = pipe.predict(ctx, prediction_length=hmax)
        # fc: list[tensor (1, 21 квантиль, hmax)] → точечный прогноз = медианный квантиль
        def to_median(f):
            arr = np.asarray(f)
            if arr.ndim == 3:
                arr = arr[0]  # (21, hmax)
            return np.median(arr, axis=0) if arr.ndim == 2 else arr
        med = np.stack([to_median(f) for f in fc])
        for h in HORIZONS:
            if h <= hmax:
                preds.append(pd.DataFrame({"series_id": sids, "origin": o, "horizon": h,
                                           "y_pred": np.expm1(np.clip(med[:, h - 1], 0, None)),
                                           "model": args.tag, "test_idx": o + h - 1}))
        print(f"origin {o}: {time.time()-t1:.0f}с ({(time.time()-t1)/len(wv)*1000:.1f} мс/ряд)", flush=True)

    pred = pd.concat(preds, ignore_index=True)
    truth = wide_rub.stack().rename("y_true").reset_index().rename(columns={"month_idx": "test_idx"})
    pred = pred.merge(truth, on=["series_id", "test_idx"], how="left")
    out = ROOT / "outputs" / f"preds_{args.tag}{'_lim' + str(args.limit) if args.limit else ''}.parquet"
    pred[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(out)
    print("saved", out)


if __name__ == "__main__":
    main()
