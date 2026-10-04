"""Сводные таблицы сравнения моделей (К2, К6) из outputs/preds_*.parquet.

Таблица T1 (главная): модель × горизонт → MAE(руб) / median nMAE / pooled RelMAE vs SNaive /
win rate vs Prophet-B / RMSE-tail-ratio. Разрезы: категория, квинтиль масштаба.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import mae, rmse  # noqa: E402

HORIZONS = [1, 3, 6, 12]


def load_all() -> pd.DataFrame:
    parts = []
    for f in sorted((ROOT / "outputs").glob("preds_*.parquet")):
        df = pd.read_parquet(f)
        if "model" not in df.columns:  # preds_stats.parquet уже длинный с model
            df["model"] = f.stem.replace("preds_", "")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def add_snaive(df: pd.DataFrame) -> pd.DataFrame:
    sn = df[df["model"] == "SeasonalNaive"][["series_id", "origin", "horizon", "y_pred"]]
    sn = sn.rename(columns={"y_pred": "y_snaive"})
    return df.merge(sn, on=["series_id", "origin", "horizon"], how="left")


def summary_by(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Сводка: MAE считается сначала по серии (равный вес МО), потом по пулу."""
    df = df.copy()
    df["ae"] = (df["y_true"] - df["y_pred"]).abs()
    df["se"] = (df["y_true"] - df["y_pred"]) ** 2
    df["ae_sn"] = (df["y_true"] - df["y_snaive"]).abs()
    rows = []
    for keys, g in df.groupby(by + ["horizon"], observed=True):
        per_series = g.groupby("series_id").agg(ae=("ae", "mean"), ae_sn=("ae_sn", "mean"),
                                                se=("se", "mean"), lvl=("y_true", "median"))
        pb = g[g["model"].isin(["Prophet-B"])].set_index(["series_id", "origin"])["y_pred"]
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row["horizon"] = keys[-1] if by else keys
        row["mae_rub"] = per_series["ae"].mean()
        row["median_nmae"] = (per_series["ae"] / per_series["lvl"]).median()
        row["relmae_vs_snaive"] = per_series["ae"].sum() / per_series["ae_sn"].sum()
        row["rmse_over_mae"] = np.sqrt(per_series["se"].mean()) / per_series["ae"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def win_rate_vs_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """Доля (series, origin)-точек, где модель точнее Prophet-B, по горизонтам."""
    wide = df.pivot_table(index=["series_id", "origin", "horizon"], columns="model",
                          values="y_pred")
    y = df.drop_duplicates(["series_id", "origin", "horizon"]).set_index(
        ["series_id", "origin", "horizon"])["y_true"]
    if "Prophet-B" not in wide.columns:
        return pd.DataFrame()
    out = {}
    for m in wide.columns:
        if m == "Prophet-B":
            continue
        out[m] = ((y - wide[m]).abs() < (y - wide["Prophet-B"]).abs()).groupby("horizon").mean()
    return pd.DataFrame(out).T.rename_axis("model").reset_index()


def main():
    df = load_all()
    df = add_snaive(df)
    # метаданные разрезов
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    meta = panel.groupby(["territory_id", "category"]).agg(
        lvl=("value", "median"), region=("region_name", "first"),
        mtype=("municipal_district_type", "first")).reset_index()
    meta["series_id"] = meta["territory_id"].astype(str) + "|" + meta["category"]
    meta["scale_q"] = pd.qcut(meta.groupby("category")["lvl"].rank(method="first"), 5,
                              labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str)
    df["category"] = df["series_id"].str.split("|").str[1]
    df = df.merge(meta[["series_id", "scale_q", "mtype"]], on="series_id", how="left")

    t1 = summary_by(df, ["model"])
    t1 = t1.sort_values(["horizon", "relmae_vs_snaive"])
    t1.to_csv(ROOT / "outputs" / "T1_model_x_horizon.csv", index=False)

    wr = win_rate_vs_prophet(df)
    if len(wr):
        wr.to_csv(ROOT / "outputs" / "T1b_winrate_vs_prophetB.csv", index=False)

    for slice_name, col in [("category", "category"), ("scale", "scale_q")]:
        t = summary_by(df, ["model", col])
        t.to_csv(ROOT / "outputs" / f"T2_slice_{slice_name}.csv", index=False)

    # консольная витрина: Все категории
    tot = df[df["category"] == "Все категории"]
    show = summary_by(tot, ["model"]).pivot(index="model", columns="horizon",
                                            values="relmae_vs_snaive").round(3)
    print("=== Все категории: pooled RelMAE vs SeasonalNaive (ниже = лучше) ===")
    print(show.to_string())


if __name__ == "__main__":
    main()
