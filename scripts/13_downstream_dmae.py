"""Downstream ΔMAE: улучшает ли refit-от-брейка прогноз? (мост CPD→прогноз, R04: решающая метрика)

Для рядов с детектированным брейком (Window@3.0, рабочая точка фронтира) в окне train:
сравнение AutoTheta на полном train vs train только после последнего брейка (fallback:
если post-break < 8 точек — полный train). Rolling-origin origins 12..23, горизонты 1/3/6.
Выход: outputs/downstream_dmae.csv (ΔMAE по горизонтам + доля рядов, где refit лучше).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from statsforecast.models import AutoTheta

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6]
ORIGINS = list(range(12, 24))
MIN_POST = 8  # post-break ряд короче → fallback на полный train (R04)


def main():
    t0 = time.time()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    panel["month_idx"] = (pd.PeriodIndex(panel["date"], freq="M").year - 2023) * 12 + \
                         pd.PeriodIndex(panel["date"], freq="M").month - 1
    wide = panel.pivot_table(index="series_id", columns="month_idx",
                             values="log_value").reindex(columns=range(24))
    wide_rub = panel.pivot_table(index="series_id", columns="month_idx",
                                 values="value").reindex(columns=range(24))
    cal = np.arange(24) % 12

    # брейки Window@3.0 на десезонированном (та же рабочая точка, что в радаре)
    rows = []
    sids = wide.index.to_numpy()
    wv, wr = wide.to_numpy(), wide_rub.to_numpy()
    n_with_break = 0
    for i, sid in enumerate(sids):
        y = wv[i]
        prof = np.array([np.nanmedian(y[cal == m]) for m in range(12)])
        y_ds = y - prof[cal]
        br = [b for b in rpt.Window(width=4, model="rbf").fit(y_ds).predict(pen=3.0) if b < 24]
        if not br:
            continue
        n_with_break += 1
        last_br = max(br)
        # величина брейка: |медиана 3 после − медиана 3 до| в единицах σ остатков
        s_ds = np.std(np.diff(y_ds)) or 1e-9
        mag = abs(np.median(y_ds[last_br:last_br+3]) - np.median(y_ds[max(0,last_br-3):last_br])) / s_ds
        for o in ORIGINS:
            if o <= last_br:  # брейк должен быть ВНУТРИ train
                continue
            for h in HORIZONS:
                tm = o - 1 + h
                if tm > 23:
                    continue
                # полный train
                try:
                    p_full = AutoTheta().fit(y[:o]).predict(h=h)["mean"][-1]
                except Exception:
                    continue
                # refit от брейка (или fallback)
                start = last_br if o - last_br >= MIN_POST else 0
                try:
                    p_br = AutoTheta().fit(y[start:o]).predict(h=h)["mean"][-1]
                except Exception:
                    p_br = p_full
                y_true = wr[i, tm]
                rows.append(dict(series_id=sid, origin=o, horizon=h,
                                 ae_full=abs(np.expm1(p_full) - y_true),
                                 ae_refit=abs(np.expm1(p_br) - y_true),
                                 post_len=o - last_br, mag=mag))
        if (i + 1) % 2000 == 0:
            print(f"{i+1}/{len(sids)}, {time.time()-t0:.0f}с", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / "outputs" / "downstream_dmae_rows.parquet")
    summ = df.groupby("horizon").apply(
        lambda g: pd.Series({
            "n": len(g),
            "mae_full": g.ae_full.mean(), "mae_refit": g.ae_refit.mean(),
            "dmae_pct": (g.ae_refit.mean() / g.ae_full.mean() - 1) * 100,
            "share_refit_better": (g.ae_refit < g.ae_full).mean()}),
        include_groups=False).reset_index()
    summ.to_csv(ROOT / "outputs" / "downstream_dmae.csv", index=False)
    df["mag_bucket"] = pd.cut(df["mag"], [0, 1, 2, 4, np.inf], labels=["<1σ", "1-2σ", "2-4σ", ">4σ"])
    by_mag = df[df.horizon == 3].groupby("mag_bucket", observed=True).apply(
        lambda g: pd.Series({"n": len(g), "dmae_pct": (g.ae_refit.mean()/g.ae_full.mean()-1)*100,
                             "share_refit_better": (g.ae_refit < g.ae_full).mean()}),
        include_groups=False).reset_index()
    by_mag.to_csv(ROOT / "outputs" / "downstream_dmae_by_mag.csv", index=False)
    print(by_mag.round(3).to_string(index=False))
    print(f"рядов с брейком: {n_with_break} из {len(sids)}")
    print(summ.round(3).to_string(index=False))
    print(f"total {time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()
