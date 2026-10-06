"""Сводные таблицы сравнения моделей (К2, К6) из outputs/preds_*.parquet.

ВЕРСИЯ 06.10 после аудита rc0 (P0-1): строго matched cells.
- На каждом горизонте берётся ПЕРЕСЕЧЕНИЕ ячеек (series_id × origin) всех моделей
  этого горизонта (кроме структурно невозможных: LGBM h=12 — 0 легальных train-строк).
- RelMAE = Σ per-series MAE / Σ per-series MAE SNaive на ТЕХ ЖЕ ячейках (B3 §2.2).
- В каждой таблице — n_origins и n_series (таблица обязана себя документировать).
- Разрезы: категория, квинтиль масштаба, тип МО, терциль волатильности (train-окно ≤ 12).
- Значимость: DM+HLN (h≤6), Wilcoxon + region-cluster bootstrap (h=12), Holm — src/significance.py.
- Win rate считается против Prophet-A И Prophet-B* (B* = честный splice, 03b).

Запуск: python scripts/04_build_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.significance import (dm_test, paired_wilcoxon, region_cluster_bootstrap_ci,  # noqa: E402
                              holm, effect_sizes)

HORIZONS = [1, 3, 6, 12]
# Старые файлы preds_stats_ces_part* (починка CES на 1-origin сетке) исключены:
# честный CES входит в preds_stats_part* после перегона 06.10 (utilsforecast 0.2.16).
EXCLUDE_GLOB = ["preds_stats_ces_part*", "preds_stats_steps_part*"]


def load_all() -> pd.DataFrame:
    parts = []
    for f in sorted((ROOT / "outputs").glob("preds_*.parquet")):
        if any(f.match(p) for p in EXCLUDE_GLOB):
            continue
        df = pd.read_parquet(f)
        if "model" not in df.columns:
            df["model"] = f.stem.replace("preds_", "")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["model"] != "y"]
    df = df.drop_duplicates(["series_id", "origin", "horizon", "model"], keep="last")
    return df


def matched_cells(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Пересечение ячеек (series_id, origin) по всем моделям внутри каждого горизонта."""
    if models:
        df = df[df["model"].isin(models)]
    keep_parts = []
    for h, g in df.groupby("horizon"):
        present = [m for m, gg in g.groupby("model") if len(gg) > 0]
        sets = [set(zip(gg["series_id"], gg["origin"])) for m, gg in g.groupby("model")]
        inter = set.intersection(*sets) if sets else set()
        sel = g[g.apply(lambda r: (r["series_id"], r["origin"]) in inter, axis=1)]
        sel = sel.copy()
        sel["n_models_matched"] = len(present)
        keep_parts.append(sel)
    return pd.concat(keep_parts, ignore_index=True)


def add_snaive(df: pd.DataFrame) -> pd.DataFrame:
    sn = df[df["model"] == "SeasonalNaive"][["series_id", "origin", "horizon", "y_pred"]]
    sn = sn.rename(columns={"y_pred": "y_snaive"})
    return df.merge(sn, on=["series_id", "origin", "horizon"], how="left")


def summary_by(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Сводка: MAE сначала по серии (равный вес МО), потом по пулу; RelMAE на тех же ячейках."""
    df = df.copy()
    df["ae"] = (df["y_true"] - df["y_pred"]).abs()
    df["se"] = (df["y_true"] - df["y_pred"]) ** 2
    df["ae_sn"] = (df["y_true"] - df["y_snaive"]).abs()
    rows = []
    for keys, g in df.groupby(by + ["horizon"], observed=True):
        per_series = g.groupby("series_id").agg(ae=("ae", "mean"), ae_sn=("ae_sn", "mean"),
                                                se=("se", "mean"), lvl=("y_true", "median"))
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row["horizon"] = keys[-1] if by else keys
        row["mae_rub"] = per_series["ae"].mean()
        row["median_nmae"] = (per_series["ae"] / per_series["lvl"]).median()
        row["relmae_vs_snaive"] = per_series["ae"].sum() / per_series["ae_sn"].sum()
        row["rmse_over_mae"] = np.sqrt(per_series["se"].mean()) / per_series["ae"].mean()
        row["n_series"] = int(per_series.shape[0])
        row["n_origins"] = int(g["origin"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def win_rate(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Доля (series, origin)-ячеек, где модель точнее baseline. Честные пары без NaN."""
    wide = df.pivot_table(index=["series_id", "origin", "horizon"], columns="model",
                          values="y_pred")
    y = df.drop_duplicates(["series_id", "origin", "horizon"]).set_index(
        ["series_id", "origin", "horizon"])["y_true"]
    if baseline not in wide.columns:
        return pd.DataFrame()
    out = {}
    for m in wide.columns:
        if m == baseline:
            continue
        a, b = (y - wide[m]).abs(), (y - wide[baseline]).abs()
        ok = a.notna() & b.notna()
        out[m] = (a[ok] < b[ok]).groupby("horizon").mean()
    res = pd.DataFrame(out).T.rename_axis("model").reset_index()
    res["baseline"] = baseline
    return res


def per_series_mae(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ae"] = (d["y_true"] - d["y_pred"]).abs()
    return d.groupby(["model", "horizon", "series_id"], as_index=False)["ae"].mean()


def significance_block(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """DM+HLN (h≤6) и Wilcoxon+bootstrap (h=12) против Prophet-B* и Prophet-A, Holm по горизонту."""
    rows = []
    d = df.copy()
    d["ae"] = (d["y_true"] - d["y_pred"]).abs()
    ps = per_series_mae(df)
    regions = meta.set_index("series_id")["region"]

    for baseline in ["Prophet-B*", "Prophet-A"]:
        if baseline not in set(d["model"]):
            continue
        for h in HORIZONS:
            dh = d[d["horizon"] == h]
            base = dh[dh["model"] == baseline].set_index(["origin", "series_id"])["ae"]
            pvals = {}
            per_model = {}
            for m, g in dh.groupby("model"):
                if m == baseline:
                    continue
                cur = g.set_index(["origin", "series_id"])["ae"]
                common = cur.index.intersection(base.index)
                if len(common) < 1000:
                    continue
                ea, eb = cur.loc[common], base.loc[common]
                if h <= 6:
                    r = dm_test(ea, eb, h)
                else:
                    r = {"stat": np.nan, "p": np.nan}
                # парный Wilcoxon по per-series MAE (все h, включая 12)
                ma = g.merge(ps[ps["model"] == m][["series_id", "ae"]], on="series_id")  # не нужно; ps уже per-series
                ma = ps[(ps["model"] == m) & (ps["horizon"] == h)].set_index("series_id")["ae"]
                mb = ps[(ps["model"] == baseline) & (ps["horizon"] == h)].set_index("series_id")["ae"]
                w = paired_wilcoxon(ma, mb)
                es = effect_sizes(ma, mb)
                boot = region_cluster_bootstrap_ci(ma, mb, regions) if h == 12 else {}
                per_model[m] = {**r, **{f"w_{k}": v for k, v in w.items()}, **es, **boot}
                pvals[m] = r.get("p", np.nan) if h <= 6 else w.get("p", np.nan)
            adj = holm(pvals)
            for m, r in per_model.items():
                rows.append({"baseline": baseline, "horizon": h, "model": m,
                             **r, "p_holm": adj.get(m, np.nan)})
    return pd.DataFrame(rows)


def main():
    print("load...", flush=True)
    df = load_all()
    print(f"  {len(df)} строк, модели: {sorted(df.model.unique())}", flush=True)

    # метаданные разрезов (всё — train-окно ≤ месяца 12, ex-ante)
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    train = panel[panel["month_idx"] <= 11]
    vol = train.groupby(["territory_id", "category"])["log_value"].apply(
        lambda s: s.diff().std()).rename("vol").reset_index()
    meta = panel.groupby(["territory_id", "category"]).agg(
        lvl=("value", "median"), region=("region_name", "first"),
        mtype=("municipal_district_type", "first")).reset_index()
    meta["series_id"] = meta["territory_id"].astype(str) + "|" + meta["category"]
    meta = meta.merge(vol, on=["territory_id", "category"], how="left")
    meta["scale_q"] = pd.qcut(meta.groupby("category")["lvl"].rank(method="first"), 5,
                              labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str)
    meta["vol_t"] = pd.qcut(meta.groupby("category")["vol"].rank(method="first"), 3,
                            labels=["V1", "V2", "V3"]).astype(str)
    df["category"] = df["series_id"].str.split("|").str[1]
    df = df.merge(meta[["series_id", "scale_q", "mtype", "vol_t"]], on="series_id", how="left")

    print("matched cells...", flush=True)
    cov = df.groupby(["model", "horizon"])["origin"].nunique().unstack()
    print("покрытие origins ДО matched:\n", cov.fillna(0).astype(int).to_string(), flush=True)
    dm_ = matched_cells(df)
    print(f"  после matched: {len(dm_)} строк", flush=True)
    dm_ = add_snaive(dm_)

    t1 = summary_by(dm_, ["model"]).sort_values(["horizon", "relmae_vs_snaive"])
    t1.to_csv(ROOT / "outputs" / "T1_model_x_horizon.csv", index=False)
    print("T1 saved", flush=True)

    for name, base in [("T1b_winrate_vs_prophetB.csv", "Prophet-B*"),
                       ("T1c_winrate_vs_prophetA.csv", "Prophet-A")]:
        wr = win_rate(dm_, base)
        if len(wr):
            wr.to_csv(ROOT / "outputs" / name, index=False)

    for slice_name, col in [("category", "category"), ("scale", "scale_q"),
                            ("motype", "mtype"), ("volatility", "vol_t")]:
        t = summary_by(dm_, ["model", col])
        t.to_csv(ROOT / "outputs" / f"T2_slice_{slice_name}.csv", index=False)

    print("significance...", flush=True)
    sig = significance_block(dm_, meta)
    sig.to_csv(ROOT / "outputs" / "T3_significance.csv", index=False)

    # консольная витрина: pooled и «Все категории»
    for label, sub in [("POOLED (все категории)", dm_), ("«Все категории»", dm_[dm_["category"] == "Все категории"])]:
        show = summary_by(sub, ["model"]).pivot(index="model", columns="horizon",
                                                values="relmae_vs_snaive").round(3)
        print(f"\n=== {label}: RelMAE vs SeasonalNaive (matched cells) ===")
        print(show.to_string(), flush=True)


if __name__ == "__main__":
    main()
