"""Cold-start блок (К4): новые МО, старт 2024-01, нет 2023 (Химки и др.).

Когорта: territory_id, у которых в raw consumption.parquet ровно 2024-01..2024-12
(9 территорий x 6 категорий = 54 ряда). Из 56 МО с ровно 12 мес только эти 9 имеют
скорябельное будущее: у 43 только 2023 (нет факта 2024), у 4 — разрозненные 12 мес.

Дизайн (вместо буквального «origin = их месяц 12», который не имеет ground truth):
контексты 3/6/9 мес (origin = их месяц L, прогноз на остаток 2024, h = 1..12-L).
Chronos-2 zero-shot (log1p) vs SeasonalNaive (при контексте < 12 мес = Naive,
полного сезонного цикла нет — в этом и суть cold-start) vs HistoricAverage.

Выход: outputs/preds_coldstart.parquet + outputs/coldstart_chronos.csv
(модель x горизонт -> MAE, RelMAE vs SeasonalNaive, win rate).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = [3, 6, 9]  # длина контекста в месяцах; origin = месяц L (0-based idx L)


def main():
    raw = pd.read_parquet(ROOT / "data" / "raw" / "hackathon" / "hackathonlicence" / "consumption.parquet")
    cnt = raw.groupby("territory_id")["date"].agg(["min", "max", "nunique"])
    cohort = cnt[(cnt["min"] == "2024-01") & (cnt["max"] == "2024-12") & (cnt["nunique"] == 12)].index
    sub = raw[raw["territory_id"].isin(cohort)].copy()
    sub["series_id"] = sub["territory_id"].astype(str) + "|" + sub["category"]
    sub["month_idx"] = pd.PeriodIndex(sub["date"], freq="M").month - 1  # 0..11 внутри 2024
    wide = sub.pivot_table(index="series_id", columns="month_idx", values="value").reindex(columns=range(12))
    assert wide.notna().all().all()
    print(f"cohort: {len(cohort)} территорий, {len(wide)} рядов", flush=True)

    from chronos import Chronos2Pipeline
    t0 = time.time()
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps")
    print(f"load {time.time()-t0:.0f}с", flush=True)

    wv = np.log1p(wide.to_numpy())
    sids = wide.index.to_numpy()
    preds = []
    for L in CONTEXTS:
        hmax = 12 - L
        ctx = [wv[i, :L] for i in range(len(wv))]
        t1 = time.time()
        fc = pipe.predict(ctx, prediction_length=hmax)
        med = np.stack([np.median(np.asarray(f)[0], axis=0) for f in fc])  # (n, hmax)
        print(f"ctx {L} мес: {time.time()-t1:.1f}с", flush=True)
        for h in range(1, hmax + 1):
            y_true = wide.to_numpy()[:, L + h - 1]
            y_hat = np.expm1(np.clip(med[:, h - 1], 0, None))
            last = wide.to_numpy()[:, L - 1]
            preds.append(pd.DataFrame({
                "series_id": sids, "ctx_len": L, "horizon": h, "y_true": y_true,
                "chronos2_zs": y_hat,
                "seasonal_naive": last,  # контекст < 12: сезонный лаг недоступен -> last value
                "historic_average": np.expm1(wv[:, :L].mean(axis=1)),
            }))
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(ROOT / "outputs" / "preds_coldstart.parquet")

    rows = []
    models = ["chronos2_zs", "seasonal_naive", "historic_average"]
    for h in sorted(pred["horizon"].unique()):
        g = pred[pred["horizon"] == h]
        mae = {m: (g["y_true"] - g[m]).abs().mean() for m in models}
        for m in models:
            win = ((g["y_true"] - g[m]).abs() < (g["y_true"] - g["seasonal_naive"]).abs()).mean()
            rows.append({"model": m, "horizon": h, "n_cells": len(g),
                         "MAE": mae[m], "RelMAE": mae[m] / mae["seasonal_naive"],
                         "win_rate_vs_snaive": win})
    out = pd.DataFrame(rows)[["model", "horizon", "n_cells", "MAE", "RelMAE", "win_rate_vs_snaive"]]
    out.to_csv(ROOT / "outputs" / "coldstart_chronos.csv", index=False)
    print(out.pivot_table(index="horizon", columns="model", values="RelMAE").round(3))
    print("saved outputs/coldstart_chronos.csv")

    # разрез по длине контекста (самая зрелищная ось: RelMAE по ctx_len, pooled по h)
    rows2 = []
    for L in CONTEXTS:
        g = pred[pred["ctx_len"] == L]
        mae = {m: (g["y_true"] - g[m]).abs().mean() for m in models}
        for m in models:
            win = ((g["y_true"] - g[m]).abs() < (g["y_true"] - g["seasonal_naive"]).abs()).mean()
            rows2.append({"model": m, "ctx_len": L, "n_cells": len(g),
                          "MAE": mae[m], "RelMAE": mae[m] / mae["seasonal_naive"], "win_rate_vs_snaive": win})
    out2 = pd.DataFrame(rows2)
    out2.to_csv(ROOT / "outputs" / "coldstart_chronos_by_ctx.csv", index=False)
    print(out2.pivot_table(index="ctx_len", columns="model", values="RelMAE").round(3))


if __name__ == "__main__":
    main()
