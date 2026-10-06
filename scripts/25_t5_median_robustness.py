"""T5: медианная устойчивость headline-побед — не «победа на крупных МО».

Для каждой заявленной победы (DriftAdjSNaive vs Prophet-B*, h=1/3/6; SeasProfileHA
vs Prophet-B*, h=12) на срезе «Все категории» (matched cells):
- median per-series RelMAE модели и базлайна (RelMAE_i = MAE_i/MAE_i(SNaive));
- медиана парного отношения MAE_i(model)/MAE_i(B*) и доля рядов, где модель лучше;
- pooled RelMAE рядом (для сравнения агрегаций).
Контрольный pooled-фрейм (все 6 категорий, 11682 ряда) — рядом.
Дополнительно: тот же блок по квинтилям масштаба Q1..Q5 (победа обязана жить
не только в Q5).

Выход: outputs/T5_median_robustness.csv
Запуск: nice -n 10 .venv/bin/python scripts/25_t5_median_robustness.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_GLOB = ["preds_stats_ces_part*", "preds_stats_steps_part*", "preds_chronos2_quant_o12*",
                "preds_chronos2_zs_raw_o12*", "preds_coldstart*", "preds_lgbm_s*"]
MODELS = ["DriftAdjSNaive", "SeasProfileHA", "Prophet-B*", "SeasonalNaive"]
PAIRS = [("DriftAdjSNaive", 1), ("DriftAdjSNaive", 3), ("DriftAdjSNaive", 6),
         ("SeasProfileHA", 12)]


def load_matched() -> pd.DataFrame:
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


def scale_quintiles() -> pd.Series:
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    meta = panel.groupby(["territory_id", "category"]).agg(lvl=("value", "median")).reset_index()
    meta["series_id"] = meta["territory_id"].astype(str) + "|" + meta["category"]
    meta["scale_q"] = pd.qcut(meta.groupby("category")["lvl"].rank(method="first"), 5,
                              labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str)
    return meta.set_index("series_id")["scale_q"]


def block(d: pd.DataFrame, model: str, h: int, frame: str, slice_name: str) -> dict:
    dh = d[d.horizon == h]
    ps = dh.groupby(["model", "series_id"])["ae"].mean().unstack("model")
    ps = ps.dropna(subset=[model, "Prophet-B*", "SeasonalNaive"])
    rel_m = ps[model] / ps["SeasonalNaive"]
    rel_b = ps["Prophet-B*"] / ps["SeasonalNaive"]
    return {"frame": frame, "slice": slice_name, "model": model, "horizon": h,
            "n_series": len(ps),
            "median_relmae_model": round(float(rel_m.median()), 4),
            "median_relmae_bstar": round(float(rel_b.median()), 4),
            "median_ratio_vs_bstar": round(float((ps[model] / ps["Prophet-B*"]).median()), 4),
            "win_rate_series_vs_bstar": round(float((ps[model] < ps["Prophet-B*"]).mean()), 4),
            "pooled_relmae_model": round(float(ps[model].sum() / ps["SeasonalNaive"].sum()), 4),
            "pooled_relmae_bstar": round(float(ps["Prophet-B*"].sum() / ps["SeasonalNaive"].sum()), 4)}


def main():
    df = load_matched()
    df["ae"] = (df["y_true"] - df["y_pred"]).abs()
    df["category"] = df["series_id"].str.split("|").str[1]
    sq = scale_quintiles()
    df["scale_q"] = df["series_id"].map(sq)

    rows = []
    d_head = df[df.category == "Все категории"]
    for model, h in PAIRS:
        rows.append(block(d_head, model, h, "headline_Все_категории", "all"))
        rows.append(block(df, model, h, "pooled_6_категорий", "all"))
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            rows.append(block(d_head[d_head.scale_q == q], model, h, "headline_Все_категории", q))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "T5_median_robustness.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
