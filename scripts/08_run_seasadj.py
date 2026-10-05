"""Два принципиальных кандидата (рационале зафиксировано ДО прогона, см. git log):

1) DriftAdjSNaive — сезонный наивный с поправкой на дрейф. Мотивация a priori (E3, 04.10):
   номинальный YoY рост +13.1% → SNaive систематически занижает 2024. Дрейф = pooled
   кросс-секционная медиана YoY-прироста на origin (per-series дрейф на ≤24 точках шумный;
   pooled устойчив и не заглядывает вперёд).
2) SeasProfileHA — HistoricAverage + per-МО сезонный профиль (шринкаж к региональному, k=2).
   Мотивация a priori: HA выигрывает h=1 регрессией к среднему, но слеп к сезонности
   (декабрь ×1.37–1.47); per-МО профиль добавляет сезонность без дрейфа.

Оба — точечные атаки на измеренные слабости скамьи, не «подбор победителя».
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6, 12]
ORIGINS = list(range(12, 24))
CAL = np.arange(24) % 12


def main():
    t0 = time.time()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    panel["month_idx"] = (pd.PeriodIndex(panel["date"], freq="M").year - 2023) * 12 + \
                         pd.PeriodIndex(panel["date"], freq="M").month - 1
    wide = panel.pivot_table(index="series_id", columns="month_idx", values="log_value").reindex(columns=range(24))
    wide_rub = panel.pivot_table(index="series_id", columns="month_idx", values="value").reindex(columns=range(24))
    wv, wr = wide.to_numpy(), wide_rub.to_numpy()
    truth = wide_rub.stack().rename("y_true").reset_index().rename(columns={"month_idx": "test_idx"})

    preds = []
    for o in ORIGINS:
        c = o - 1
        train_cols = list(range(c + 1))
        med_all = np.nanmedian(wv[:, train_cols], axis=1)
        # pooled YoY дрейф на origin (по всем рядам; месяцы c-11..c против c-23..c-12)
        if c >= 12:
            yoy_series = wv[:, c] - wv[:, c - 12]
            g = np.nanmedian(yoy_series)
        else:
            g = 0.0
        for h in HORIZONS:
            tm = c + h
            if tm > 23:
                continue
            # 1) drift-adjusted seasonal naive
            if tm - 12 >= 0:
                p_log = wv[:, tm - 12] + g
            else:
                p_log = med_all + g
            preds.append(pd.DataFrame({"series_id": wide.index, "origin": o, "horizon": h,
                                       "y_pred": np.expm1(np.clip(p_log, 0, None)),
                                       "model": "DriftAdjSNaive", "test_idx": tm}))
            # 2) HA + сезонный профиль (шринкаж к региональному, k=2)
            mcols = [i for i in train_cols if CAL[i] == tm % 12]
            mo_seas = np.nanmedian(wv[:, mcols], axis=1) - med_all if mcols else 0.0
            reg = panel.groupby("series_id")["region_name"].first()
            reg_ratio = (pd.DataFrame(wv[:, mcols]).groupby(reg.values).median().median(axis=1)
                         - pd.DataFrame(wv[:, train_cols]).groupby(reg.values).median().median(axis=1)
                         ).reindex(reg.values).to_numpy() if mcols else np.zeros(len(wide))
            k_shrink = 2.0
            seas = (len(mcols) * np.nan_to_num(mo_seas, nan=0.0) + k_shrink * reg_ratio) / (len(mcols) + k_shrink)
            p_log2 = med_all + seas
            preds.append(pd.DataFrame({"series_id": wide.index, "origin": o, "horizon": h,
                                       "y_pred": np.expm1(np.clip(p_log2, 0, None)),
                                       "model": "SeasProfileHA", "test_idx": tm}))
    pred = pd.concat(preds, ignore_index=True).merge(truth, on=["series_id", "test_idx"], how="left")
    pred[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(
        ROOT / "outputs" / "preds_seasadj.parquet", index=False)
    print(f"saved preds_seasadj.parquet, {len(pred)} строк, {time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()
