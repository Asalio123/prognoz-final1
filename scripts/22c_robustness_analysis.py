"""Сборка робастности B2 (всё из готовых preds; новые прогоны — только 22/22b).

Выходы:
- outputs/robustness_sliding18.csv   — sliding18 vs expanding(origins 18+) на идентичных ячейках
- outputs/robustness_per_origin.csv  — h=1 «Все категории»: MAE по 12 origin'ам + win-rate
- outputs/robustness_filters.csv     — h=1: без топ-5% крупных МО / без Общепита / только Q1-Q2
- outputs/robustness_bootstrap_h1.csv— region-cluster bootstrap CI разности медиан
                                       DriftAdjSNaive vs Prophet-B* (и vs Prophet-A), «Все категории»

Агрегация идентична scripts/04_build_tables.py: per-series MAE → mean по пулу;
RelMAE = Σ per-series MAE / Σ per-series MAE SNaive на тех же ячейках.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.significance import paired_wilcoxon, region_cluster_bootstrap_ci, effect_sizes  # noqa: E402

CHEAP = ["SeasonalNaive", "DynamicOptimizedTheta", "AutoETS", "DriftAdjSNaive"]
PROPHETS = ["Prophet-A", "Prophet-B*"]
SLIDING_ORIGINS = {1: list(range(18, 24)), 3: list(range(18, 22))}


def load_expanding() -> pd.DataFrame:
    """Expanding preds ключевых моделей (stats chunks + seasadj + prophet A/B*)."""
    parts = []
    for f in sorted((ROOT / "outputs").glob("preds_stats_part*.parquet")):
        parts.append(pd.read_parquet(f))
    for f in ["preds_seasadj.parquet", "preds_prophet_A.parquet", "preds_prophet_Bfixed.parquet"]:
        parts.append(pd.read_parquet(ROOT / "outputs" / f))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["model"].isin(CHEAP + PROPHETS)]
    return df.drop_duplicates(["series_id", "origin", "horizon", "model"], keep="last")


def load_sliding() -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted((ROOT / "outputs").glob("robustness_sliding18_part*.parquet"))]
    for f in sorted((ROOT / "outputs").glob("robustness_sliding18_prophet*.parquet")):
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def per_series_ae(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ae"] = (d["y_true"] - d["y_pred"]).abs()
    return d.groupby(["model", "horizon", "series_id"], as_index=False).agg(
        ae=("ae", "mean"), lvl=("y_true", "median"))


def summarize(df: pd.DataFrame, label_cols: dict) -> pd.DataFrame:
    """Сводка по df (уже подготовленному подмножеству одной схемы/среза/горизонта)."""
    ps = per_series_ae(df)
    sn = ps[ps["model"] == "SeasonalNaive"].set_index("series_id")["ae"]
    rows = []
    for (m, h), g in ps.groupby(["model", "horizon"]):
        g = g.dropna(subset=["ae"])
        common = g["series_id"].map(lambda s: s in sn.index)
        g = g[common]
        denom = sn.loc[g["series_id"]].sum()
        rows.append({**label_cols, "model": m, "horizon": h,
                     "mae_rub": g["ae"].mean(),
                     "median_nmae": (g["ae"] / g["lvl"]).median(),
                     "relmae_vs_snaive": g["ae"].sum() / denom if denom else np.nan,
                     "n_series": int(g.shape[0]),
                     "n_origins": int(df[(df.model == m) & (df.horizon == h)]["origin"].nunique())})
    return pd.DataFrame(rows)


def intersect_cells(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Пересечение (series_id, origin) ячеек по моделям внутри каждого горизонта."""
    keep = []
    for h, g in df.groupby("horizon"):
        sets = [set(zip(gg["series_id"], gg["origin"]))
                for m, gg in g.groupby("model") if m in models]
        inter = set.intersection(*sets)
        sel = g[g.apply(lambda r: (r["series_id"], r["origin"]) in inter, axis=1)]
        keep.append(sel)
    return pd.concat(keep, ignore_index=True)


def task1_sliding(exp: pd.DataFrame, sli: pd.DataFrame) -> pd.DataFrame:
    """Sliding18 vs expanding на origins 18+ (идентичные ячейки)."""
    exp_r = pd.concat([exp[(exp.horizon == h) & (exp.origin.isin(origs))]
                       for h, origs in SLIDING_ORIGINS.items()], ignore_index=True)
    # контроль построения: DriftAdj и SNaive sliding == expanding на этих ячейках
    for m in ["DriftAdjSNaive", "SeasonalNaive"]:
        a = sli[sli.model == m].set_index(["series_id", "origin", "horizon"])["y_pred"].sort_index()
        b = exp_r[exp_r.model == m].set_index(["series_id", "origin", "horizon"])["y_pred"].sort_index()
        common = a.index.intersection(b.index)
        rel = (a.loc[common] - b.loc[common]).abs() / b.loc[common].clip(lower=1)
        print(f"  контроль {m}: max rel отличие sliding vs expanding = {rel.max():.2e} "
              f"(ожидание ~0, окно 18 >= 13 мес памяти)", flush=True)

    out = []
    for scope, e_sub, s_sub, models in [
        ("pooled", exp_r, sli, CHEAP),
        ("Все категории", exp_r[exp_r.series_id.str.endswith("|Все категории")],
         sli[sli.series_id.str.endswith("|Все категории")], CHEAP + PROPHETS),
    ]:
        e_m = intersect_cells(e_sub[e_sub.model.isin(models)], models)
        s_m = intersect_cells(s_sub[s_sub.model.isin(models)], models)
        out.append(summarize(e_m, {"scheme": "expanding_o18+", "scope": scope}))
        out.append(summarize(s_m, {"scheme": "sliding18", "scope": scope}))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(ROOT / "outputs" / "robustness_sliding18.csv", index=False)
    return res


def task2_per_origin(exp: pd.DataFrame) -> pd.DataFrame:
    """Per-origin профиль h=1 «Все категории»: чемпион vs Prophet-B* vs SNaive."""
    models = ["DriftAdjSNaive", "Prophet-B*", "Prophet-A", "SeasonalNaive"]
    d = exp[(exp.horizon == 1) & (exp.model.isin(models)) &
            (exp.series_id.str.endswith("|Все категории"))].copy()
    d["ae"] = (d["y_true"] - d["y_pred"]).abs()
    mae = d.groupby(["model", "origin"])["ae"].mean().unstack("model")
    mae["relmae_drift"] = mae["DriftAdjSNaive"] / mae["SeasonalNaive"]
    mae["relmae_prophetB"] = mae["Prophet-B*"] / mae["SeasonalNaive"]
    mae["drift_beats_prophetB"] = mae["DriftAdjSNaive"] < mae["Prophet-B*"]
    # per-series win share чемпиона против B* внутри origin'а
    wide = d.pivot_table(index=["series_id", "origin"], columns="model", values="ae")
    win = (wide["DriftAdjSNaive"] < wide["Prophet-B*"]).groupby("origin").mean()
    mae["series_win_share_drift_vs_B"] = win
    mae = mae.reset_index()
    mae.to_csv(ROOT / "outputs" / "robustness_per_origin.csv", index=False)
    wr = mae["drift_beats_prophetB"].mean()
    print(f"\n  win-rate DriftAdj vs Prophet-B* по origin'ам: {mae['drift_beats_prophetB'].sum()}/12 = {wr:.2f}")
    print(f"  медианная per-series win share: {mae['series_win_share_drift_vs_B'].median():.3f}")
    return mae


def task3_filters(exp: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Чувствительность главного результата (h=1) к фильтрам панели."""
    models = ["DriftAdjSNaive", "Prophet-B*", "Prophet-A", "SeasonalNaive",
              "DynamicOptimizedTheta", "AutoETS"]
    d = exp[(exp.horizon == 1) & (exp.model.isin(models))].copy()
    # масштаб МО — train-окно (месяцы 0..11), ex-ante; квинтили внутри категории
    tr = panel[panel["month_idx"] <= 11]
    lvl = tr.groupby(["territory_id", "category"])["value"].median().rename("lvl_train").reset_index()
    lvl["series_id"] = lvl["territory_id"].astype(str) + "|" + lvl["category"]
    lvl["scale_q"] = lvl.groupby("category")["lvl_train"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]))
    d = d.merge(lvl[["series_id", "lvl_train", "scale_q"]], on="series_id", how="left")

    variants = {
        "full_ВсеКатегории": d[d.series_id.str.endswith("|Все категории")],
        "no_top5_ВсеКатегории": d[d.series_id.str.endswith("|Все категории") &
                                  (d.scale_q != "Q5")],
        "no_obschepit_pooled": d[d.series_id.str.split("|").str[1] != "Общественное питание"],
        "small_Q1Q2_ВсеКатегории": d[d.series_id.str.endswith("|Все категории") &
                                     d.scale_q.isin(["Q1", "Q2"])],
    }
    out = []
    for name, sub in variants.items():
        sub = intersect_cells(sub, models)
        s = summarize(sub, {"filter": name})
        b = s[s.model == "Prophet-B*"].set_index("horizon")["mae_rub"]
        s["mae_ratio_vs_prophetB"] = s.apply(
            lambda r: r["mae_rub"] / b.get(r["horizon"], np.nan), axis=1)
        out.append(s)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(ROOT / "outputs" / "robustness_filters.csv", index=False)
    return res


def task4_bootstrap(exp: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Region-cluster bootstrap CI разности медиан per-series MAE, «Все категории», h=1."""
    d = exp[(exp.horizon == 1) & (exp.series_id.str.endswith("|Все категории"))].copy()
    d["ae"] = (d["y_true"] - d["y_pred"]).abs()
    ps = d.groupby(["model", "series_id"], as_index=False)["ae"].mean()
    reg = panel.assign(series_id=panel["territory_id"].astype(str) + "|" + panel["category"])
    regions = reg.groupby("series_id")["region_name"].first()

    rows = []
    for other in ["Prophet-B*", "Prophet-A"]:
        ma = ps[ps.model == "DriftAdjSNaive"].set_index("series_id")["ae"]
        mb = ps[ps.model == other].set_index("series_id")["ae"]
        common = ma.index.intersection(mb.index)
        ma, mb = ma.loc[common], mb.loc[common]
        boot = region_cluster_bootstrap_ci(ma, mb, regions, n_boot=10_000, seed=42)
        w = paired_wilcoxon(ma, mb)
        es = effect_sizes(ma, mb)
        rows.append({"comparison": f"DriftAdjSNaive - {other}", "horizon": 1,
                     "scope": "Все категории", "n_series": len(common),
                     **boot, "wilcoxon_p": w["p"], **es})
        print(f"\n  {rows[-1]['comparison']}: median diff {boot['median_diff']:.1f} руб, "
              f"95% CI [{boot['ci_lo']:.1f}, {boot['ci_hi']:.1f}], wilcoxon p={w['p']:.2e}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "robustness_bootstrap_h1.csv", index=False)
    return res


def main():
    print("load expanding...", flush=True)
    exp = load_expanding()
    print("load sliding...", flush=True)
    sli = load_sliding()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")

    print("\n== задача 1: sliding18 ==", flush=True)
    t1 = task1_sliding(exp, sli)
    show = t1.pivot_table(index=["scope", "model"], columns="scheme",
                          values="relmae_vs_snaive")
    print(show.round(4).to_string(), flush=True)

    print("\n== задача 2: per-origin ==", flush=True)
    t2 = task2_per_origin(exp)
    print(t2.round(1).to_string(), flush=True)

    print("\n== задача 3: фильтры ==", flush=True)
    t3 = task3_filters(exp, panel)
    print(t3[t3.model.isin(["DriftAdjSNaive", "Prophet-B*", "SeasonalNaive"])]
          .round(4).to_string(), flush=True)

    print("\n== задача 4: bootstrap ==", flush=True)
    task4_bootstrap(exp, panel)


if __name__ == "__main__":
    main()
