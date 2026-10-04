"""Глобальный LightGBM (К2): direct-per-horizon, objective=l1 (прогноз медианы под MAE).

Фичи (все — прошлое относительно cutoff, информационное множество строго ≤ cutoff):
- лаги log_value [1,2,3,6,12], MoM/YoY приросты, скользящие средние 3/6
- календарь целевого месяца (сезонность), месяц cutoff
- per-series статика: медианный уровень (log), std MoM
- кросс-секция: региональный LOO-прирост (E3: корр. внутри региона 0.43) и LOO-уровень
- market_access (статика МО), тип МО, регион (categorical)
Валидация: configs/validation.yaml (expanding origins 12..23, горизонты 1/3/6/12).
Таргет log1p, оценка в рублях после expm1. Сид фиксирован (configs).
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HORIZONS = [1, 3, 6, 12]
ORIGINS = list(range(12, 24))
LAGS = [1, 2, 3, 6, 12]
SEED = 42


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Широкая матрица log_value по месяцам + фичи на (series_id, cutoff)."""
    panel = panel.copy()
    panel["month_idx"] = (pd.PeriodIndex(panel["date"], freq="M").year - 2023) * 12 + \
                         pd.PeriodIndex(panel["date"], freq="M").month - 1
    wide = panel.pivot_table(index="series_id", columns="month_idx", values="log_value")
    wide = wide.reindex(columns=range(24))
    meta = panel.groupby("series_id").agg(
        category=("category", "first"), region=("region_name", "first"),
        mtype=("municipal_district_type", "first")).reset_index()

    # региональный LOO-прирост по месяцам: (сумма региона − собственный) / (n−1)
    mom = wide.diff(axis=1)
    reg = panel.groupby("series_id")["region_name"].first()
    reg_mom_sum = mom.groupby(reg).sum()
    reg_mom_cnt = mom.groupby(reg).count()
    reg_lvl_sum = wide.groupby(reg).sum()
    reg_lvl_cnt = wide.groupby(reg).count()

    rows = []
    sids = wide.index.to_numpy()
    cat = meta.set_index("series_id")["category"]
    mtype = meta.set_index("series_id")["mtype"]
    for c in range(1, 24):  # cutoff = последний наблюдаемый месяц (0-indexed)
        base = pd.DataFrame(index=wide.index)
        ok = pd.Series(True, index=wide.index)
        for k in LAGS:
            base[f"lag{k}"] = wide[c + 1 - k] if c + 1 - k >= 0 else np.nan
        base["roll3"] = wide[[max(0, c - 2), max(0, c - 1), c]].mean(axis=1)
        base["roll6"] = wide[[max(0, c - 5 + i) for i in range(6)]].mean(axis=1)
        base["mom1"] = wide[c] - wide[c - 1]
        base["yoy"] = wide[c] - wide[c - 12] if c >= 12 else np.nan
        base["lvl_med"] = wide[[i for i in range(c + 1)]].median(axis=1)
        base["mom_std"] = mom[[i for i in range(1, c + 1)]].std(axis=1) if c >= 2 else np.nan
        # LOO-региональные
        r = reg
        loo_num = reg_mom_sum[c].reindex(r.values).to_numpy() - mom[c].to_numpy()
        loo_den = (reg_mom_cnt[c].reindex(r.values).to_numpy() - 1).clip(min=1)
        base["region_loo_mom"] = loo_num / loo_den
        loo_lvl_num = reg_lvl_sum[c].reindex(r.values).to_numpy() - wide[c].to_numpy()
        base["region_loo_lvl"] = loo_lvl_num / loo_den
        base["cutoff"] = c
        base["cutoff_month"] = (c % 12) + 1
        for h in [1, 3, 6, 12]:
            base[f"target_month_h{h}"] = ((c + h) % 12) + 1
        base["category"] = cat
        base["mtype"] = mtype
        base["series_id"] = sids
        rows.append(base.reset_index(drop=True))
    feats = pd.concat(rows, ignore_index=True)
    return feats, wide, meta


def main():
    t0 = time.time()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    feats, wide, meta = build_features(panel)
    feats["category"] = feats["category"].astype("category")
    feats["mtype"] = feats["mtype"].astype("category")
    print(f"фичи: {feats.shape}, {time.time()-t0:.0f}с")

    feat_cols = [f"lag{k}" for k in LAGS] + ["roll3", "roll6", "mom1", "yoy", "lvl_med",
                                             "mom_std", "region_loo_mom", "region_loo_lvl",
                                             "cutoff_month", "category", "mtype"]
    # векторные таргеты: y_mat[h][i, c] = log_value ряда i в месяц c+h
    sid_pos = {s: i for i, s in enumerate(wide.index)}
    wvals = wide.to_numpy()
    y_mat = {h: np.hstack([wvals[:, h:], np.full((wvals.shape[0], h), np.nan)]) for h in HORIZONS}
    feats["sidx"] = feats["series_id"].map(sid_pos).to_numpy()

    preds = []
    for o in ORIGINS:
        cutoff = o - 1  # последний наблюдаемый месяц
        for h in HORIZONS:
            test_m = cutoff + h
            if test_m > 23:
                continue
            tr_mask = (feats["cutoff"] + h <= cutoff) & (feats["cutoff"] >= max(1, cutoff - 11))
            tr = feats[tr_mask]
            y_tr = y_mat[h][tr["sidx"].to_numpy(), tr["cutoff"].to_numpy()]
            ok = ~np.isnan(y_tr)
            te = feats[feats["cutoff"] == cutoff]
            if ok.sum() < 1000 or len(te) == 0:
                continue
            fc = feat_cols + [f"target_month_h{h}"]
            dtrain = lgb.Dataset(tr.loc[ok, fc], label=y_tr[ok],
                                 categorical_feature=["category", "mtype"], free_raw_data=False)
            params = dict(objective="l1", learning_rate=0.08, num_leaves=63,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                          seed=SEED, verbose=-1, num_threads=8)
            m = lgb.train(params, dtrain, num_boost_round=400,
                          valid_sets=[dtrain], callbacks=[lgb.log_evaluation(0)])
            p = m.predict(te[fc])
            preds.append(pd.DataFrame({"series_id": te["series_id"].values, "origin": o,
                                       "horizon": h, "y_pred": np.expm1(np.clip(p, 0, None))}))
        print(f"origin {o} готов, {time.time()-t0:.0f}с", flush=True)

    pred = pd.concat(preds, ignore_index=True)
    truth = panel[["series_id", "month_idx", "value"]].drop_duplicates()
    truth = truth.rename(columns={"value": "y_true", "month_idx": "test_idx"})
    pred["test_idx"] = pred["origin"] + pred["horizon"] - 1
    pred = pred.merge(truth, on=["series_id", "test_idx"], how="left")
    pred = pred.assign(model="LightGBM-direct")
    pred[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(
        ROOT / "outputs" / "preds_lgbm.parquet")
    print(f"saved outputs/preds_lgbm.parquet, {len(pred)} строк, всего {time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()
