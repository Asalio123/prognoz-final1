"""rc1 / O3: абляция энсембля ±TSFM (К4, «влияние на качество»).

Вопрос рубрики: даёт ли добавление TSFM (Chronos-2 / TimesFM-2.5) в энсембль
прирост против лучшей одиночной модели на matched cells.

Дизайн (все ячейки = пересечение series_id × origin участников внутри горизонта,
как scripts/04_build_tables.py):
  (а) Chronos-2-zs + DriftAdjSNaive 50/50
  (б) Chronos-2-cov + DriftAdjSNaive 50/50
  (в) TimesFM-2.5-zs + DriftAdjSNaive 50/50
  (г) inverse-MAE веса, обучены ТОЛЬКО на origins 12..19 (первые 8), оценка на 20..23
  (д) Chronos-2-cov + DriftAdjSNaive + Prophet-B* (равные веса)
Гейт (configs/decision.json, тот же смысл что у Ensemble-median3): включение ⟺
ΔMAE ≤ −1% против лучшей одиночной на тех же ячейках.
Значимость: win rate по ячейкам + парный Wilcoxon по per-series MAE (src/significance.py).

Запуск: .venv/bin/python scripts/rc1_o3_tsfm_ensemble.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.significance import paired_wilcoxon  # noqa: E402

HORIZONS = [1, 3, 6, 12]
FIT_ORIGINS = list(range(12, 20))   # первые 8 из 12 — обучение весов (г)
EVAL_ORIGINS = list(range(20, 24))  # последние 4 — честная оценка (г)
GATE = -0.01                        # ΔMAE ≤ −1% против лучшей одиночной

# одиночные-кандидаты на «лучшую одиночную» (чемпионы К2 + TSFM-компоненты)
SINGLES = ["DriftAdjSNaive", "SeasProfileHA", "Prophet-B*",
           "chronos2_zs_log", "chronos2_cov_log", "timesfm25_zs_log"]

BLENDS = {
    "blend_chronosZS+DriftAdj_50/50": (["chronos2_zs_log", "DriftAdjSNaive"], None),
    "blend_chronosCOV+DriftAdj_50/50": (["chronos2_cov_log", "DriftAdjSNaive"], None),
    "blend_timesfm+DriftAdj_50/50": (["timesfm25_zs_log", "DriftAdjSNaive"], None),
    "blend_chronosCOV+DriftAdj+ProphetB*_eq": (
        ["chronos2_cov_log", "DriftAdjSNaive", "Prophet-B*"], None),
}
INV_MAE_PAIRS = {
    "w_chronosZS+DriftAdj_invMAE": ["chronos2_zs_log", "DriftAdjSNaive"],
    "w_chronosCOV+DriftAdj_invMAE": ["chronos2_cov_log", "DriftAdjSNaive"],
    "w_timesfm+DriftAdj_invMAE": ["timesfm25_zs_log", "DriftAdjSNaive"],
}


def load_preds() -> pd.DataFrame:
    parts = []
    for f, models in [
        ("preds_chronos2_zs_log.parquet", None),
        ("preds_chronos2_cov_log.parquet", None),
        ("preds_timesfm25_zs_log.parquet", None),
        ("preds_seasadj.parquet", None),
        ("preds_prophet_Bfixed.parquet", None),
    ]:
        df = pd.read_parquet(ROOT / "outputs" / f)
        if models:
            df = df[df["model"].isin(models)]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["origin"] = df["origin"].astype(int)
    df = df.drop_duplicates(["series_id", "origin", "horizon", "model"], keep="last")
    df["category"] = df["series_id"].str.split("|").str[1]
    return df


def matched_wide(df: pd.DataFrame, models: list[str], h: int,
                 origins: list[int] | None = None) -> pd.DataFrame:
    """wide-таблица (series_id, origin) × models на matched cells одного горизонта."""
    d = df[(df["horizon"] == h) & (df["model"].isin(models))]
    if origins is not None:
        d = d[d["origin"].isin(origins)]
    wide = d.pivot_table(index=["series_id", "origin", "category"],
                         columns="model", values="y_pred")
    y = d.drop_duplicates(["series_id", "origin"]).set_index(
        ["series_id", "origin", "category"])["y_true"]
    wide["y_true"] = y
    wide = wide.dropna(subset=models + ["y_true"])
    return wide.reset_index()


def eval_block(wide: pd.DataFrame, models: list[str], blend_pred: pd.Series,
               blend_name: str, h: int, tag: str) -> tuple[dict, pd.Series]:
    """Метрики бленда и всех singles на ОДНИХ ячейках. Возвращает строки + AE бленда."""
    ae = {m: (wide["y_true"] - wide[m]).abs() for m in models if m in wide}
    ae[blend_name] = (wide["y_true"] - blend_pred).abs()
    all_models = [m for m in dict.fromkeys(list(ae.keys()))]
    rows = {}
    df_ae = pd.DataFrame(ae)
    df_ae["series_id"] = wide["series_id"].values
    per_series = {m: g for m, g in
                  df_ae.groupby("series_id").mean(numeric_only=True).items()}
    per_series = {m: s.dropna() for m, s in per_series.items()}
    for m in all_models:
        rows[m] = {"mae_rub": per_series[m].mean(), "n_series": int(per_series[m].shape[0]),
                   "n_origins": int(wide.reset_index()["origin"].nunique())}
    # лучшая одиночная на этих ячейках
    singles = [m for m in all_models if m != blend_name]
    best_single = min(singles, key=lambda m: rows[m]["mae_rub"])
    b_mae = rows[blend_name]["mae_rub"]
    s_mae = rows[best_single]["mae_rub"]
    d_mae = b_mae / s_mae - 1.0
    # win rate по ячейкам против лучшей одиночной
    win = (ae[blend_name] < ae[best_single]).mean()
    # парный Wilcoxon по per-series MAE
    w = paired_wilcoxon(per_series[blend_name], per_series[best_single])
    row = {"tag": tag, "horizon": h, "blend": blend_name,
           "mae_blend": b_mae, "best_single": best_single, "mae_best_single": s_mae,
           "delta_mae_pct": 100 * d_mae, "gate_pass": bool(d_mae <= GATE),
           "win_rate_vs_best": float(win), "wilcoxon_p": w["p"],
           "wilcoxon_n": w["n"], "median_diff_rub": w["median_diff"],
           "n_series": rows[blend_name]["n_series"], "n_origins": rows[blend_name]["n_origins"],
           "mae_by_model": {m: rows[m]["mae_rub"] for m in all_models}}
    return row, ae[blend_name]


def main():
    print("load...", flush=True)
    df = load_preds()
    print(f"  {len(df)} строк; модели: {sorted(df.model.unique())}", flush=True)

    results, weights_log = [], []

    for h in HORIZONS:
        for name, (models, _) in BLENDS.items():
            wide = matched_wide(df, models, h)
            bp = wide[models].mean(axis=1)
            for scope, sub, bp_sub in [
                ("pooled", wide, bp),
                ("allcat", wide[wide["category"] == "Все категории"],
                 bp[wide["category"] == "Все категории"]),
            ]:
                if len(sub) == 0:
                    continue
                row, _ = eval_block(sub, models, bp_sub, name, h, scope)
                results.append(row)

        # (г) inverse-MAE: веса на origins 12..19, оценка на 20..23
        # сетка 12/10/7/1: доступные origins зависят от горизонта (h=12 → только origin 12,
        # holdout не существует — честно пропускаем)
        avail = set(df[df["horizon"] == h]["origin"].unique())
        fit_o = [o for o in FIT_ORIGINS if o in avail]
        ev_o = [o for o in EVAL_ORIGINS if o in avail]
        for name, models in INV_MAE_PAIRS.items():
            if not fit_o or not ev_o:
                weights_log.append({"horizon": h, "blend": name, "skipped": "no_holdout"})
                continue
            fit = matched_wide(df, models, h, origins=fit_o)
            mae_fit = {m: (fit["y_true"] - fit[m]).abs().mean() for m in models}
            w = {m: 1.0 / v for m, v in mae_fit.items()}
            s = sum(w.values())
            w = {m: v / s for m, v in w.items()}
            weights_log.append({"horizon": h, "blend": name, **w})
            ev = matched_wide(df, models, h, origins=ev_o)
            bp = sum(ev[m] * w[m] for m in models)
            row, _ = eval_block(ev, models, bp, name, h, "pooled_holdout_o20-23")
            results.append(row)
            sub = ev[ev["category"] == "Все категории"]
            if len(sub):
                row2, _ = eval_block(sub, models, bp[sub.index], name, h,
                                     "allcat_holdout_o20-23")
                results.append(row2)

    res = pd.DataFrame(results)
    res.to_csv(ROOT / "outputs" / "rc1_o3_tsfm_ensemble.csv", index=False)
    pd.DataFrame(weights_log).to_csv(ROOT / "outputs" / "rc1_o3_invmae_weights.csv", index=False)

    show = res[["tag", "horizon", "blend", "mae_blend", "best_single",
                "mae_best_single", "delta_mae_pct", "gate_pass", "win_rate_vs_best",
                "wilcoxon_p", "n_series", "n_origins"]]
    pd.set_option("display.width", 220)
    print("\n=== rc1/O3: TSFM в энсембле (matched cells) ===")
    print(show.round(4).to_string(index=False), flush=True)
    print("\nВеса inverse-MAE (fit origins 12..19):")
    print(pd.DataFrame(weights_log).round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
