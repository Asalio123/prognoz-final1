"""23. Split-conformal интервалы для чемпионов (DriftAdjSNaive, SeasProfileHA).

Дизайн (обоснование честности):
- Калибровка = inner-origin (expanding-window) остатки: для тестового origin o
  используются ТОЛЬКО ошибки на месяцах t <= o-1 (внутри обучающего окна),
  где прогноз месяца t перевычислен as-of (данные <= t-h). Это out-of-sample
  ошибки той же процедуры, без утечки. In-sample остатки НЕ используются.
- Конформность в log-пространстве: модели живут в log (см. 08_run_seasadj),
  |log1p(y) - log1p(yhat)| ~ относительная ошибка -> нормализация масштабом
  не нужна, интервалы автоматически гетероскедастичны по рядам.
- Пулы per (модель, горизонт, origin, категория, календарный месяц цели)
  с tiered fallback:
    T1: anchor-режим (t>=12), тот же календарный месяц
    T2: любой режим, тот же календарный месяц   (ловит декабрь: t=11)
    T3: anchor-режим, все месяцы
    T4: любой режим, все месяцы
    T5: кросс-горизонт, тот же календарный месяц (для o=12,h=12)
    T6: кросс-горизонт, все месяцы
  Честная цена: декабрь-2023 (t=11) — единственный in-sample декабрь и он
  до-anchor-режима => декабрьские интервалы консервативны (раскрыто в отчёте);
  origin 12 целиком калибруется на fallback-режиме (t<=11) => консервативен.
- Квантиль конформальный: k = ceil((n+1)*(1-alpha)), симметричные полосы
  вокруг точечного прогноза в log, обратно в рубли через expm1, lo>=0.

Выходы:
  outputs/intervals_champion.parquet  (series_id, origin, horizon, model,
                                       y_pred, lo80, hi80, lo95, hi95, tier)
  outputs/intervals_coverage.csv      (покрытие 80/95 + ширины, срезы:
                                       overall / origin / cal_month /
                                       category / scale_quintile)
Плюс печать: сравнение с Chronos-2 на matched origin-12 ячейках и
флаг «рискованный месяц» (ширина 80% PI > p80 по ряду): декабрь vs среднее.
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
CHAMPION_OF_H = {1: "DriftAdjSNaive", 3: "DriftAdjSNaive",
                 6: "DriftAdjSNaive", 12: "SeasProfileHA"}


# ---------------------------------------------------------------- модели as-of
def drift_pred_log(wv, t, h):
    """DriftAdjSNaive: прогноз месяца t по данным <= t-h (точная копия 08)."""
    c = t - h
    if c < 0:
        return None
    med_all = np.nanmedian(wv[:, :c + 1], axis=1)
    g = np.nanmedian(wv[:, c] - wv[:, c - 12]) if c >= 12 else 0.0
    if t - 12 >= 0:
        return wv[:, t - 12] + g
    return med_all + g


def seasprofile_pred_log(wv, reg_values, t, h):
    """SeasProfileHA: прогноз месяца t по данным <= t-h (точная копия 08)."""
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


# ---------------------------------------------------------------- конформал
def conf_quant(scores, alpha):
    """Конформальный квантиль |остатков|: k = ceil((n+1)*(1-alpha))."""
    s = np.sort(np.asarray(scores, dtype=float))
    s = s[~np.isnan(s)]
    n = len(s)
    if n == 0:
        return np.nan, 0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return s[k - 1], n


def wilson(p, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def select_pool(res_by_h, o, h, target_cal):
    """Tiered выбор калибровочного пула. Возвращает (t_list, h_list, tier)."""
    t_max = o - 1
    # T1: anchor-режим, тот же календарный месяц
    ts = [t for t in range(12, t_max + 1) if t % 12 == target_cal and t - h >= 0]
    if ts:
        return ts, [h] * len(ts), 1
    # T2: любой режим, тот же календарный месяц
    ts = [t for t in range(h, t_max + 1) if t % 12 == target_cal]
    if ts:
        return ts, [h] * len(ts), 2
    # T3: anchor-режим, все месяцы
    ts = [t for t in range(max(12, h), t_max + 1)]
    if ts:
        return ts, [h] * len(ts), 3
    # T4: любой режим, все месяцы, тот же h
    ts = [t for t in range(h, t_max + 1)]
    if ts:
        return ts, [h] * len(ts), 4
    # T5: кросс-горизонт, тот же календарный месяц
    pairs = [(t, hp) for hp in HORIZONS for t in range(hp, t_max + 1) if t % 12 == target_cal]
    if pairs:
        return [p[0] for p in pairs], [p[1] for p in pairs], 5
    # T6: кросс-горизонт, все месяцы
    pairs = [(t, hp) for hp in HORIZONS for t in range(hp, t_max + 1)]
    if pairs:
        return [p[0] for p in pairs], [p[1] for p in pairs], 6
    return [], [], 0


def build_residuals(wv, reg_values):
    """RES[model][h] = матрица (n_series x 24) |log-остатков| as-of; NaN где нет."""
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
    reg = panel.groupby("series_id")["region_name"].first().reindex(series_ids)
    reg_values = reg.to_numpy()

    res, pred = build_residuals(wv, reg_values)

    # --- регрессионный чек: as-of прогнозы == preds_seasadj.parquet ---
    ref = pd.read_parquet(ROOT / "outputs" / "preds_seasadj.parquet")
    h_map = {h: i for i, h in enumerate(HORIZONS)}
    max_diff = 0.0
    for h in HORIZONS:
        for o in ORIGINS:
            tm = o + h - 1
            if tm > 23:
                continue
            for mdl in res:
                mine = np.expm1(np.clip(pred[mdl][h][:, tm], 0, None))
                sub = ref[(ref.model == mdl) & (ref.origin == o) & (ref.horizon == h)]
                sub = sub.set_index("series_id").reindex(series_ids)
                d = np.nanmax(np.abs(mine - sub["y_pred"].to_numpy()))
                max_diff = max(max_diff, d)
    assert max_diff < 1e-6, f"as-of прогнозы не совпали с preds_seasadj: {max_diff}"
    print(f"[check] as-of прогнозы == preds_seasadj (max |diff| = {max_diff:.2e})")

    # --- интервалы ---
    rows = []
    pool_stats = []
    cats = np.unique(cat_of_series)
    for mdl in res:
        for h in HORIZONS:
            for o in ORIGINS:
                tm = o + h - 1
                if tm > 23:
                    continue
                target_cal = tm % 12
                ts, hs, tier = select_pool(res[mdl], o, h, target_cal)
                y_log = pred[mdl][h][:, tm]
                for cat in cats:
                    m = cat_of_series == cat
                    if not ts:
                        q80 = q95 = np.nan
                        n_pool = 0
                    else:
                        pool = np.concatenate([res[mdl][hp][np.ix_(m, [t])].ravel()
                                               for t, hp in zip(ts, hs)])
                        pool = pool[~np.isnan(pool)]
                        q80, n_pool = conf_quant(pool, 0.20)
                        q95, _ = conf_quant(pool, 0.05)
                    pool_stats.append({"model": mdl, "horizon": h, "origin": o,
                                       "category": cat, "tier": tier, "n_pool": n_pool,
                                       "q80_log": q80, "q95_log": q95})
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
    iv.to_parquet(ROOT / "outputs" / "intervals_champion.parquet", index=False)
    ps = pd.DataFrame(pool_stats)
    ps.to_csv(ROOT / "outputs" / "intervals_pools.csv", index=False)
    print(f"[save] intervals_champion.parquet: {len(iv)} строк; "
          f"tiers: {ps.tier.value_counts().sort_index().to_dict()}")

    # --- покрытие на тестовых ячейках ---
    truth = wide_rub.stack().rename("y_true").reset_index().rename(columns={"month_idx": "tm"})
    iv["tm"] = iv["origin"] + iv["horizon"] - 1
    df = iv.merge(truth, on=["series_id", "tm"], how="left")
    df["category"] = df["series_id"].str.split("|").str[1]
    df["cal_month"] = df["tm"] % 12
    df["cov80"] = (df["y_true"] >= df["lo80"]) & (df["y_true"] <= df["hi80"])
    df["cov95"] = (df["y_true"] >= df["lo95"]) & (df["y_true"] <= df["hi95"])
    df["w80"] = df["hi80"] - df["lo80"]
    df["w95"] = df["hi95"] - df["lo95"]
    scale = pd.Series(np.nanmedian(wr, axis=1), index=series_ids)
    qs = pd.qcut(scale, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    df["scale_quintile"] = df["series_id"].map(qs).astype(str)

    def cov_row(sub, mdl, h, stype, sval):
        n = len(sub)
        c80, c95 = sub["cov80"].mean(), sub["cov95"].mean()
        lo80ci, hi80ci = wilson(c80, n)
        lo95ci, hi95ci = wilson(c95, n)
        return {"model": mdl, "horizon": h, "slice_type": stype, "slice_value": sval,
                "n_cells": n, "cov80": c80, "cov95": c95,
                "cov80_lo": lo80ci, "cov80_hi": hi80ci,
                "cov95_lo": lo95ci, "cov95_hi": hi95ci,
                "med_w80_rub": sub["w80"].median(), "med_w95_rub": sub["w95"].median(),
                "med_relw80": (sub["w80"] / sub["y_pred"]).median()}

    cov_rows = []
    for mdl in res:
        for h in HORIZONS:
            sub = df[(df.model == mdl) & (df.horizon == h)]
            if not len(sub):
                continue
            cov_rows.append(cov_row(sub, mdl, h, "overall", "all"))
            for o, g in sub.groupby("origin"):
                cov_rows.append(cov_row(g, mdl, h, "origin", o))
            for cm, g in sub.groupby("cal_month"):
                cov_rows.append(cov_row(g, mdl, h, "cal_month", cm))
            for c, g in sub.groupby("category"):
                cov_rows.append(cov_row(g, mdl, h, "category", c))
            for q, g in sub.groupby("scale_quintile"):
                cov_rows.append(cov_row(g, mdl, h, "scale_quintile", q))
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(ROOT / "outputs" / "intervals_coverage.csv", index=False)
    print(f"[save] intervals_coverage.csv: {len(cov)} строк")

    # --- сравнение с Chronos-2 (его сетка = origin 12) ---
    chr_cov = pd.read_csv(ROOT / "outputs" / "chronos_quantile_coverage.csv")
    print("\n[Chronos-2 vs чемпионы] coverage 80% PI на origin 12 (matched):")
    for h in HORIZONS:
        ch = chr_cov[chr_cov.horizon == h]["coverage_80PI"].iloc[0]
        mine_rows = [cov_row(df[(df.model == mdl) & (df.horizon == h) & (df.origin == 12)],
                             mdl, h, "", "")[ "cov80"] for mdl in
                     ["DriftAdjSNaive", "SeasProfileHA"]]
        print(f"  h={h:2d}: Chronos {ch:.3f} | DriftAdj {mine_rows[0]:.3f} | "
              f"SeasProfileHA {mine_rows[1]:.3f}")

    # --- флаг «рискованный месяц» (чемпион-of-record по горизонту) ---
    ch_rows = []
    for h, mdl in CHAMPION_OF_H.items():
        ch_rows.append(df[(df.model == mdl) & (df.horizon == h)])
    chd = pd.concat(ch_rows, ignore_index=True)
    thr = chd.groupby("series_id")["w80"].quantile(0.80).rename("thr80")
    chd = chd.merge(thr, on="series_id")
    chd["flag"] = chd["w80"] > chd["thr80"]
    cell_dec = chd[chd.tm == 23]["flag"].mean()
    cell_all = chd["flag"].mean()
    mo = chd.groupby(["series_id", "tm"])["flag"].any().reset_index()
    mo_dec = mo[mo.tm == 23]["flag"].mean()
    mo_all = mo["flag"].mean()
    n_dec_mo = int(mo[mo.tm == 23]["flag"].sum())
    n_mo_months_dec = int((mo.tm == 23).sum())
    print(f"\n[Рискованный месяц] ячейки: декабрь {cell_dec:.1%} vs в среднем {cell_all:.1%}")
    print(f"[Рискованный месяц] МО-месяцы: декабрь {mo_dec:.1%} "
          f"({n_dec_mo} из {n_mo_months_dec}) vs в среднем {mo_all:.1%}")
    per_month = mo.groupby("tm")["flag"].mean()
    print("[Рискованный месяц] доля флагов по месяцам 2024:",
      {int(k): round(float(v), 3) for k, v in per_month.items()})

    print(f"\nготово за {time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()
