"""LGBM-раннер для абляции N2/N3 новостного слоя (К5; предрегистрация R07 §5 / D3 §6).

N0 = 05_run_lgbm.py --news 0          (есть: outputs/preds_lgbm.parquet)
N1 = 05_run_lgbm.py --news 1 (v2)     (есть: outputs/preds_lgbm_news_v2.parquet)
N2 = этот скрипт --config N2: N1 + внутримесячные взвешивания eom7/exp
     (outputs/news_monthly_v3.parquet, скрипт 25a; θ=7d полураспад предзадан)
N3 = этот скрипт --config N3: N2 + региональные взаимодействия L_{i,k} x NF_k:
     - ix_prices_foodshare  : share_prices_inflation x доля продовольствия МО
     - ix_rate_lvl          : share_rate_credit x средний уровень расходов МО
                              (прокси дохода/«среднего чека»: транзакционной разбивки
                              в панели нет, R07 T3 допускает train-окно прокси)
     - ix_sanctions_nonfood : share_sanctions_shocks x (1 - доля продовольствия)
     Нагрузки ТОЛЬКО из train-окна: expanding-кумулянты по месяцам <= cutoff,
     нормировка mean=1 по кросс-секции МО на каждом cutoff (пересчёт per origin).

Фича-конструктор и тренировочный цикл — дословно из 05_run_lgbm.py (импорт, не копия):
те же сид, параметры, сетка origin'ов; отличие от N1 только в наборе новостных колонок.

Выход: outputs/preds_lgbm_news{config}_v3.parquet (v2-файлы не перезаписываются).
Запуск: nice -n 10 .venv/bin/python scripts/25_run_lgbm_N23.py --config N2
"""
import importlib.util
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("run_lgbm_base", ROOT / "scripts" / "05_run_lgbm.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

HORIZONS = base.HORIZONS
ORIGINS = base.ORIGINS
LAGS = base.LAGS

TOPICS = ["prices_inflation", "rate_credit", "sanctions_shocks", "emergencies_regional"]
NEWS_V2_COLS = [f"{p}_{t}" for t in TOPICS for p in ("share", "z")]
NEWS_N2_EXTRA = [f"share_{t}_{a}" for t in TOPICS for a in ("eom7", "exp")]
IX_COLS = ["ix_prices_foodshare", "ix_rate_lvl", "ix_sanctions_nonfood"]


def build_loadings(panel: pd.DataFrame) -> pd.DataFrame:
    """Expanding-нагрузки L_{i,c} per (territory_id, cutoff): только месяцы <= cutoff.

    Возвращает длинный df: territory_id, cutoff, L_food, L_lvl, L_nonfood (mean=1
    по кросс-секции МО на каждом cutoff).
    """
    cats = panel.pivot_table(index="territory_id", columns=["month_idx", "category"],
                             values="value", aggfunc="first")
    food = cats.xs("Продовольствие", level="category", axis=1).reindex(columns=range(24))
    total = cats.xs("Все категории", level="category", axis=1).reindex(columns=range(24))
    cum_food = food.cumsum(axis=1, skipna=True)
    cum_total = total.cumsum(axis=1, skipna=True)
    n_months = pd.Series(range(1, 25), index=range(24))  # делитель для среднего уровня

    rows = []
    terrs = food.index.to_numpy()
    for c in range(24):
        fs = (cum_food[c] / cum_total[c].replace(0, np.nan)).to_numpy()
        lvl = (cum_total[c] / n_months[c]).to_numpy()
        nf = 1 - fs
        # нормировка mean=1 по кросс-секции этого cutoff (только наблюдаемые МО)
        def norm(x):
            m = np.nanmean(x)
            return x / m if m > 0 else np.full_like(x, np.nan)
        rows.append(pd.DataFrame({"territory_id": terrs, "cutoff": c,
                                  "L_food": norm(fs), "L_lvl": norm(lvl),
                                  "L_nonfood": norm(nf)}))
    return pd.concat(rows, ignore_index=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["N2", "N3"], required=True)
    ap.add_argument("--tag", default="_v3")
    ap.add_argument("--seed", type=int, default=base.SEED)
    args = ap.parse_args()
    t0 = time.time()

    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    feats, wide, meta = base.build_features(panel)
    feats["category"] = feats["category"].astype("category")
    feats["mtype"] = feats["mtype"].astype("category")
    print(f"фичи: {feats.shape}, {time.time()-t0:.0f}с", flush=True)

    sid_pos = {s: i for i, s in enumerate(wide.index)}
    wvals = wide.to_numpy()
    y_mat = {h: np.hstack([wvals[:, h:], np.full((wvals.shape[0], h), np.nan)]) for h in HORIZONS}
    feats["sidx"] = feats["series_id"].map(sid_pos).to_numpy()

    # --- новостной блок: v3 (mean из v2 переиспользован + eom7/exp из 25a)
    nm = pd.read_parquet(ROOT / "outputs" / "news_monthly_v3.parquet").reset_index()
    nm["midx"] = (pd.PeriodIndex(nm["month"], freq="M").year - 2023) * 12 + \
                 pd.PeriodIndex(nm["month"], freq="M").month - 1
    news_cols = NEWS_V2_COLS + NEWS_N2_EXTRA
    feats = feats.merge(nm[["midx"] + news_cols], left_on="cutoff", right_on="midx",
                        how="left").drop(columns="midx")
    feats[news_cols] = feats[news_cols].fillna(0.0)
    print(f"news v3 ({args.config}): нац. колонок {len(news_cols)}", flush=True)

    ix_cols = []
    if args.config == "N3":
        loads = build_loadings(panel)
        feats["territory_id"] = feats["series_id"].str.split("|").str[0].astype(loads["territory_id"].dtype)
        feats = feats.merge(loads, on=["territory_id", "cutoff"], how="left")
        # у МО без истории <= cutoff быть не должно (панель полная); страховка — нейтральная 1
        feats[["L_food", "L_lvl", "L_nonfood"]] = feats[["L_food", "L_lvl", "L_nonfood"]].fillna(1.0)
        feats["ix_prices_foodshare"] = feats["L_food"] * feats["share_prices_inflation"]
        feats["ix_rate_lvl"] = feats["L_lvl"] * feats["share_rate_credit"]
        feats["ix_sanctions_nonfood"] = feats["L_nonfood"] * feats["share_sanctions_shocks"]
        ix_cols = IX_COLS
        print(f"N3: +{len(ix_cols)} взаимодействий (L_food/lvl/nonfood mean=1 per cutoff)", flush=True)

    feat_cols = [f"lag{k}" for k in LAGS] + ["roll3", "roll6", "mom1", "yoy", "lvl_med",
                                             "mom_std", "region_loo_mom", "region_loo_lvl",
                                             "lag1_vs_yearavg", "cutoff_month", "category", "mtype"] \
        + news_cols + ix_cols
    preds = []
    for o in ORIGINS:
        cutoff = o - 1
        for h in HORIZONS:
            test_m = cutoff + h
            if test_m > 23:
                continue
            tr_mask = (feats["cutoff"] + h <= cutoff) & (feats["cutoff"] >= 1)
            tr = feats[tr_mask]
            y_tr = y_mat[h][tr["sidx"].to_numpy(), tr["cutoff"].to_numpy()]
            ok = ~np.isnan(y_tr)
            te = feats[feats["cutoff"] == cutoff]
            if ok.sum() < 1000 or len(te) == 0:
                continue
            fc = feat_cols + [f"target_month_h{h}", f"seas_target_h{h}"]
            dtrain = lgb.Dataset(tr.loc[ok, fc], label=y_tr[ok],
                                 categorical_feature=["category", "mtype"], free_raw_data=False)
            params = dict(objective="l1", learning_rate=0.08, num_leaves=63,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                          seed=args.seed, verbose=-1, num_threads=8)
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
    pred = pred.assign(model=f"LightGBM-direct-news{args.config}{args.tag}")
    out = ROOT / "outputs" / f"preds_lgbm_news{args.config}{args.tag}.parquet"
    pred[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(out)
    print(f"saved {out.relative_to(ROOT)}, {len(pred)} строк, всего {time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()
