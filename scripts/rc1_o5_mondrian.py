"""rc1/O5: Mondrian (групповой) split-conformal по масштабу ряда.

Гипотеза: недопокрытие малых МО (Q1 cov80 = 0.734 на h=1) вызвано тем, что
пул остатков per-категория смешивает ряды разного масштаба: |log-остатки|
малых МО систематически больше (низкая база -> высокая относительная
волатильность), и единый квантиль по категории их недопокрывает.

Варианты калибровки (пулы строго из прошлого каждого origin, схема 23_*):
  V0 = baseline (пулы per категория, как в 23_champion_intervals.py)
  V1 = Mondrian по квинтилю масштаба (пулы per квинтиль, категории смешаны)
  V2 = Mondrian категория x квинтиль (пулы per (категория, квинтиль))

Квинтиль масштаба: медиана рублёвого значения ряда за месяцы 0..11 (2023) —
строго раньше любой тестовой цели (min tm = 12), утечки нет. Совпадение с
квинтилями по полной истории (как в intervals_coverage.csv) проверяется.

Выходы:
  outputs/rc1_o5_coverage_by_variant.csv  — coverage/ширины по вариантам
  outputs/rc1_o5_quintile_agreement.csv   — согласие разметки квинтилей
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6, 12]
ORIGINS = list(range(12, 24))
CAL = np.arange(24) % 12
CHAMPION_OF_H = {1: "DriftAdjSNaive", 3: "DriftAdjSNaive",
                 6: "DriftAdjSNaive", 12: "SeasProfileHA"}


# --- as-of модели: точные копии из 23_champion_intervals.py ------------------
def drift_pred_log(wv, t, h):
    c = t - h
    if c < 0:
        return None
    med_all = np.nanmedian(wv[:, :c + 1], axis=1)
    g = np.nanmedian(wv[:, c] - wv[:, c - 12]) if c >= 12 else 0.0
    if t - 12 >= 0:
        return wv[:, t - 12] + g
    return med_all + g


def seasprofile_pred_log(wv, reg_values, t, h):
    c = t - h
    if c < 0:
        return None
    train_cols = list(range(c + 1))
    med_all = np.nanmedian(wv[:, train_cols], axis=1)
    mcols = [i for i in train_cols if CAL[i] == t % 12]
    if mcols:
        mo_seas = np.nanmedian(wv[:, mcols], axis=1) - med_all
        reg_ratio = (pd.DataFrame(wv[:, mcols]).groupby(reg_values).median().median(axis=1)
                     - pd.DataFrame(wv[:, train_cols]).groupby(reg_values).median().median(axis=1)
                     ).reindex(reg_values).to_numpy()
    else:
        mo_seas, reg_ratio = 0.0, np.zeros(wv.shape[0])
    k_shrink = 2.0
    seas = (len(mcols) * np.nan_to_num(mo_seas, nan=0.0) + k_shrink * reg_ratio) / (len(mcols) + k_shrink)
    return med_all + seas


def conf_quant(scores, alpha):
    s = np.sort(np.asarray(scores, dtype=float))
    s = s[~np.isnan(s)]
    n = len(s)
    if n == 0:
        return np.nan, 0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return s[k - 1], n


def select_pool(o, h):
    t_max = o - 1
    ts = [t for t in range(max(12, h), t_max + 1)]
    if ts:
        return ts, [h] * len(ts), 1
    ts = [t for t in range(h, t_max + 1)]
    if ts:
        return ts, [h] * len(ts), 2
    pairs = [(t, hp) for hp in HORIZONS for t in range(hp, t_max + 1)]
    if pairs:
        return [p[0] for p in pairs], [p[1] for p in pairs], 3
    return [], [], 0


def build_residuals(wv, reg_values):
    res = {"DriftAdjSNaive": {}, "SeasProfileHA": {}}
    pred = {"DriftAdjSNaive": {}, "SeasProfileHA": {}}
    for h in HORIZONS:
        for mdl, fn in (("DriftAdjSNaive", drift_pred_log),
                        ("SeasProfileHA", seasprofile_pred_log)):
            R = np.full((wv.shape[0], 24), np.nan)
            P = np.full((wv.shape[0], 24), np.nan)
            for t in range(h, 24):
                pl = fn(wv, t, h) if mdl == "DriftAdjSNaive" else fn(wv, reg_values, t, h)
                if pl is None:
                    continue
                P[:, t] = pl
                R[:, t] = np.abs(wv[:, t] - pl)
            res[mdl][h], pred[mdl][h] = R, P
    return res, pred


# --- группировки -------------------------------------------------------------
def group_key(variant, cat_arr, quint_arr):
    if variant == "V0_category":
        return cat_arr
    if variant == "V1_quintile":
        return quint_arr
    if variant == "V2_cat_x_quintile":
        return np.array([f"{c}|{q}" for c, q in zip(cat_arr, quint_arr)])
    raise ValueError(variant)


def main():
    t0 = time.time()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    panel["month_idx"] = (pd.PeriodIndex(panel["date"], freq="M").year - 2023) * 12 + \
                         pd.PeriodIndex(panel["date"], freq="M").month - 1
    wide = panel.pivot_table(index="series_id", columns="month_idx", values="log_value").reindex(columns=range(24))
    wide_rub = panel.pivot_table(index="series_id", columns="month_idx", values="value").reindex(columns=range(24))
    wv, wr = wide.to_numpy(), wide_rub.to_numpy()
    series_ids = wide.index.to_numpy()
    cat_of_series = pd.Series(series_ids).str.split("|").str[1].to_numpy()
    reg_values = panel.groupby("series_id")["region_name"].first().reindex(series_ids).to_numpy()

    # --- квинтили масштаба: по 2023 (месяцы 0..11, строго до всех tm>=12) ----
    scale_2023 = pd.Series(np.nanmedian(wr[:, :12], axis=1), index=series_ids)
    qs_2023 = pd.qcut(scale_2023, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    scale_full = pd.Series(np.nanmedian(wr, axis=1), index=series_ids)
    qs_full = pd.qcut(scale_full, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    agree = (qs_2023.astype(str) == qs_full.astype(str))
    adj = pd.crosstab(qs_2023.astype(str), qs_full.astype(str))
    print(f"[квинтили] согласие 2023-only vs full-history: {agree.mean():.3f} "
          f"({int(agree.sum())}/{len(agree)})")
    pd.DataFrame({"metric": ["agreement_2023_vs_full"],
                  "value": [agree.mean()]}).to_csv(
        ROOT / "outputs" / "rc1_o5_quintile_agreement.csv", index=False)
    print(adj)
    quint_arr = qs_2023.reindex(series_ids).astype(str).to_numpy()

    res, pred = build_residuals(wv, reg_values)

    truth = wide_rub.stack().rename("y_true").reset_index().rename(columns={"month_idx": "tm"})

    variants = ["V0_category", "V1_quintile", "V2_cat_x_quintile"]
    all_iv = {}
    for variant in variants:
        rows = []
        for mdl in res:
            for h in HORIZONS:
                for o in ORIGINS:
                    tm = o + h - 1
                    if tm > 23:
                        continue
                    ts, hs, tier = select_pool(o, h)
                    y_log = pred[mdl][h][:, tm]
                    gkey = group_key(variant, cat_of_series, quint_arr)
                    for g in np.unique(gkey):
                        m = gkey == g
                        if not ts:
                            q80 = q95 = np.nan
                        else:
                            pool = np.concatenate([res[mdl][hp][np.ix_(m, [t])].ravel()
                                                   for t, hp in zip(ts, hs)])
                            pool = pool[~np.isnan(pool)]
                            q80, _ = conf_quant(pool, 0.20)
                            q95, _ = conf_quant(pool, 0.05)
                        yp = np.expm1(np.clip(y_log[m], 0, None))
                        if np.isnan(q80):
                            lo80 = hi80 = lo95 = hi95 = np.full(m.sum(), np.nan)
                        else:
                            lo80 = np.expm1(np.clip(y_log[m] - q80, 0, None))
                            hi80 = np.expm1(y_log[m] + q80)
                            lo95 = np.expm1(np.clip(y_log[m] - q95, 0, None))
                            hi95 = np.expm1(y_log[m] + q95)
                        rows.append(pd.DataFrame({
                            "series_id": series_ids[m], "origin": o, "horizon": h,
                            "model": mdl, "y_pred": yp, "lo80": lo80, "hi80": hi80,
                            "lo95": lo95, "hi95": hi95, "tier": tier}))
        iv = pd.concat(rows, ignore_index=True)
        iv["tm"] = iv["origin"] + iv["horizon"] - 1
        df = iv.merge(truth, on=["series_id", "tm"], how="left")
        df["category"] = df["series_id"].str.split("|").str[1]
        df["quintile"] = df["series_id"].map(qs_2023).astype(str)
        df["cov80"] = (df["y_true"] >= df["lo80"]) & (df["y_true"] <= df["hi80"])
        df["cov95"] = (df["y_true"] >= df["lo95"]) & (df["y_true"] <= df["hi95"])
        df["w80"] = df["hi80"] - df["lo80"]
        df["relw80"] = df["w80"] / df["y_pred"]
        all_iv[variant] = df
        print(f"[build] {variant}: {len(df)} ячеек ({time.time()-t0:.0f}с)")

    # --- сводка coverage/ширин ----------------------------------------------
    def crow(variant, sub, mdl, h, stype, sval):
        return {"variant": variant, "model": mdl, "horizon": h,
                "slice_type": stype, "slice_value": sval, "n_cells": len(sub),
                "cov80": sub["cov80"].mean(), "cov95": sub["cov95"].mean(),
                "med_w80_rub": sub["w80"].median(),
                "med_relw80": sub["relw80"].median()}

    cov_rows = []
    for variant, df in all_iv.items():
        for mdl in res:
            for h in HORIZONS:
                sub = df[(df.model == mdl) & (df.horizon == h)]
                if not len(sub):
                    continue
                cov_rows.append(crow(variant, sub, mdl, h, "overall", "all"))
                sub1 = sub[sub.tier == 1]
                if len(sub1):
                    cov_rows.append(crow(variant, sub1, mdl, h, "overall_tier1", "anchor_only"))
                for q, g in sub.groupby("quintile"):
                    cov_rows.append(crow(variant, g, mdl, h, "quintile", q))
                dec = sub[sub.tm == 23]
                if len(dec):
                    cov_rows.append(crow(variant, dec, mdl, h, "december_tm23", "dec"))
        # champion-of-record: pooled по горизонтам
        chd = pd.concat([df[(df.model == mdl) & (df.horizon == h)]
                         for h, mdl in CHAMPION_OF_H.items()], ignore_index=True)
        cov_rows.append(crow(variant, chd, "CHAMPION", "pooled", "overall", "all"))
        cov_rows.append(crow(variant, chd[chd.tier == 1], "CHAMPION", "pooled",
                             "overall_tier1", "anchor_only"))
        for q, g in chd.groupby("quintile"):
            cov_rows.append(crow(variant, g, "CHAMPION", "pooled", "quintile", q))
        dec = chd[chd.tm == 23]
        cov_rows.append(crow(variant, dec, "CHAMPION", "pooled", "december_tm23", "dec"))
        # декабрь по квинтилям (риск: Mondrian по масштабу vs сезонный риск)
        for q, g in dec.groupby("quintile"):
            cov_rows.append(crow(variant, g, "CHAMPION", "pooled", "december_x_quintile", q))
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(ROOT / "outputs" / "rc1_o5_coverage_by_variant.csv", index=False)
    print(f"[save] outputs/rc1_o5_coverage_by_variant.csv: {len(cov)} строк")

    # --- печать ключевых сравнений -------------------------------------------
    pd.set_option("display.width", 250)
    for scope, sub0 in [("CHAMPION pooled", cov[cov.model == "CHAMPION"])]:
        print(f"\n=== {scope}: overall_tier1 ===")
        print(sub0[sub0.slice_type == "overall_tier1"]
              [["variant", "n_cells", "cov80", "cov95", "med_w80_rub", "med_relw80"]]
              .to_string(index=False))
        print(f"\n=== {scope}: по квинтилям (все ячейки) ===")
        print(sub0[sub0.slice_type == "quintile"]
              [["variant", "slice_value", "n_cells", "cov80", "med_w80_rub", "med_relw80"]]
              .to_string(index=False))
        print(f"\n=== {scope}: декабрь tm=23 ===")
        print(sub0[sub0.slice_type == "december_tm23"]
              [["variant", "n_cells", "cov80", "med_relw80"]].to_string(index=False))
        print(f"\n=== {scope}: декабрь x квинтиль ===")
        print(sub0[sub0.slice_type == "december_x_quintile"]
              [["variant", "slice_value", "cov80", "med_relw80"]].to_string(index=False))

    # рост ширины vs V0 (pooled champion, tier1)
    base = cov[(cov.model == "CHAMPION") & (cov.slice_type == "overall_tier1")].set_index("variant")
    print("\n=== рост медианной ширины относительно V0 (champion tier1) ===")
    for v in variants:
        print(f"  {v}: med_w80={base.loc[v,'med_w80_rub']:.0f} ₽ "
              f"({base.loc[v,'med_w80_rub']/base.loc['V0_category','med_w80_rub']-1:+.1%}), "
              f"med_relw80={base.loc[v,'med_relw80']:.3f} "
              f"({base.loc[v,'med_relw80']/base.loc['V0_category','med_relw80']-1:+.1%})")

    # DriftAdj h=1 по квинтилям (прямое сравнение с 0.734 из отчёта)
    print("\n=== DriftAdjSNaive h=1 по квинтилям (срез из отчёта, все ячейки) ===")
    d1 = cov[(cov.model == "DriftAdjSNaive") & (cov.horizon == 1) & (cov.slice_type == "quintile")]
    print(d1[["variant", "slice_value", "cov80", "med_w80_rub"]].to_string(index=False))

    print(f"\nготово за {time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()
