"""Медианный энсембль (предрегистрирован в configs/champion.yaml: members_frozen_after=first_full_run).

Состав (структурно разнородные): AutoTheta + DynamicOptimizedTheta + LightGBM-direct.
Медиана по трём — робастна к выбросам одного члена (M4/M5-урок: комбинации > одиночные).
Гейт включения: ΔMAE ≤ −1% против лучшей одиночной модели на co-primary горизонтах.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ["AutoTheta", "DynamicOptimizedTheta", "LightGBM-direct"]
NAME = "Ensemble-median3"


def main():
    parts = []
    for f in sorted((ROOT / "outputs").glob("preds_*.parquet")):
        df = pd.read_parquet(f)
        parts.append(df[df["model"].isin(MEMBERS)])
    df = pd.concat(parts, ignore_index=True)
    ens = (df.groupby(["series_id", "origin", "horizon"], as_index=False)
             .agg(y_pred=("y_pred", "median"), y_true=("y_true", "first")))
    ens["model"] = NAME
    ens.to_parquet(ROOT / "outputs" / "preds_ensemble.parquet", index=False)
    print(f"saved preds_ensemble.parquet, {len(ens)} строк")


if __name__ == "__main__":
    main()
