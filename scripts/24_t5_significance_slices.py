"""T5: разрезная значимость headline-побед на срезе «Все категории» (matched cells).

Протокол configs/validation.yaml + src/significance.py:
- h ∈ {1,3,6}: DM + Newey-West(lag=h−1) + HLN по per-origin loss differentials
  (кросс-секция = 2016 рядов категории «Все категории») + парный Wilcoxon.
- h = 12 (1 origin): DM невозможен по построению → парный Wilcoxon по 2016 МО
  + 95% CI разности медиан region-cluster bootstrap (10 000 ресэмплов).
- Holm внутри семейства из 4 headline-тестов.
Пары (заявленные победы): DriftAdjSNaive vs Prophet-B* на h=1/3/6;
SeasProfileHA vs Prophet-B* на h=12. Контрольный pooled-фрейм (все 6 категорий)
считается рядом — чтобы зафиксировать, где победа живёт, а где нет.

Выход: outputs/T5_significance_slices.csv
Запуск: nice -n 10 .venv/bin/python scripts/24_t5_significance_slices.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.significance import (dm_test, paired_wilcoxon, region_cluster_bootstrap_ci,  # noqa: E402
                              holm, effect_sizes)

EXCLUDE_GLOB = ["preds_stats_ces_part*", "preds_stats_steps_part*", "preds_chronos2_quant_o12*",
                "preds_chronos2_zs_raw_o12*", "preds_coldstart*", "preds_lgbm_s*"]
MODELS = ["DriftAdjSNaive", "SeasProfileHA", "Prophet-B*", "SeasonalNaive"]
PAIRS = [("DriftAdjSNaive", 1), ("DriftAdjSNaive", 3), ("DriftAdjSNaive", 6),
         ("SeasProfileHA", 12)]


def load_matched() -> pd.DataFrame:
    """Matched cells ровно как 04_build_tables.py (пересечение 20 моделей T1); y_pred — только MODELS."""
    key_sets, keep = {}, []
    files = [f for f in sorted((ROOT / "outputs").glob("preds_*.parquet"))
             if not any(f.match(p) for p in EXCLUDE_GLOB)]
    for f in files:
        df = pd.read_parquet(f)
        if "model" not in df.columns:
            df["model"] = f.stem.replace("preds_", "")
        df = df[df["model"] != "y"]
        df = df.drop_duplicates(["series_id", "origin", "horizon", "model"], keep="last")
        for (m, h), g in df.groupby(["model", "horizon"]):
            key_sets.setdefault(h, {}).setdefault(m, set()).update(zip(g["series_id"], g["origin"]))
        if df["model"].isin(MODELS).any():
            keep.append(df[df["model"].isin(MODELS)])
    inter = {h: set.intersection(*s.values()) for h, s in key_sets.items()}
    df = pd.concat(keep, ignore_index=True)
    return df[df.apply(lambda r: (r["series_id"], r["origin"]) in inter[r["horizon"]], axis=1)]


def run_frame(d: pd.DataFrame, regions: pd.Series, frame: str) -> list[dict]:
    rows = []
    pvals = {}
    for model, h in PAIRS:
        dh = d[d.horizon == h]
        ea = dh[dh.model == model].set_index(["origin", "series_id"])["ae"]
        eb = dh[dh.model == "Prophet-B*"].set_index(["origin", "series_id"])["ae"]
        common = ea.index.intersection(eb.index)
        ea, eb = ea.loc[common], eb.loc[common]
        # per-series MAE
        ma = ea.groupby("series_id").mean()
        mb = eb.groupby("series_id").mean()
        r = {"frame": frame, "model": model, "horizon": h, "baseline": "Prophet-B*",
             "n_cells": len(common), "n_series": ma.shape[0],
             "n_origins": int(dh["origin"].nunique())}
        if h <= 6:
            r.update({f"dm_{k}": v for k, v in dm_test(ea, eb, h).items()})
        w = paired_wilcoxon(ma, mb)
        r.update({f"w_{k}": v for k, v in w.items()})
        r.update({f"es_{k}": v for k, v in effect_sizes(ma, mb).items()})
        if h == 12:
            r.update({f"boot_{k}": v for k, v in region_cluster_bootstrap_ci(ma, mb, regions).items()})
        pvals[(frame, model, h)] = r.get("dm_p", np.nan) if h <= 6 else w.get("p", np.nan)
        rows.append(r)
    adj = holm({f"{f}|{m}|{h}": p for (f, m, h), p in pvals.items()})
    for r in rows:
        r["p_holm_family4"] = adj.get(f"{r['frame']}|{r['model']}|{r['horizon']}", np.nan)
    return rows


def main():
    df = load_matched()
    df["ae"] = (df["y_true"] - df["y_pred"]).abs()
    df["category"] = df["series_id"].str.split("|").str[1]
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    regions = panel.groupby("series_id")["region_name"].first()

    rows = []
    rows += run_frame(df[df.category == "Все категории"], regions, "headline_Все_категории")
    rows += run_frame(df, regions, "pooled_6_категорий")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "T5_significance_slices.csv", index=False)
    cols = ["frame", "model", "horizon", "n_cells", "n_origins", "dm_stat", "dm_p",
            "w_p", "es_median_ratio", "boot_ci_lo", "boot_ci_hi", "p_holm_family4"]
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
