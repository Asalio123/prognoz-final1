"""T5: per-category чемпионы — «категория × горизонт → лучшая модель + RelMAE + отрыв от Prophet-B*».

Источник: outputs/T2_slice_category.csv (matched cells, те же числа, что в T1/T2/T3).
Отрыв: delta_relmae = RelMAE(Prophet-B*) − RelMAE(лучшая) (>0 = лучшая бьёт B*),
ratio_vs_bstar = RelMAE(лучшая)/RelMAE(Prophet-B*) (<1 = бьёт).
Маркетплейсы — отдельный блок: топ-5 на каждом горизонте + проверка тренд-гипотезы
числами из панели (pooled median YoY по категориям, train-окно).

Запуск: nice -n 10 .venv/bin/python scripts/23_t5_category_champions.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6, 12]


def main():
    t2 = pd.read_csv(ROOT / "outputs" / "T2_slice_category.csv")
    bstar = t2[t2.model == "Prophet-B*"].set_index(["category", "horizon"])["relmae_vs_snaive"]

    rows = []
    for (cat, h), g in t2.groupby(["category", "horizon"]):
        best = g.loc[g["relmae_vs_snaive"].idxmin()]
        b = bstar.get((cat, h), np.nan)
        rows.append({"category": cat, "horizon": h, "champion": best["model"],
                     "relmae": round(best["relmae_vs_snaive"], 4),
                     "mae_rub": round(best["mae_rub"], 1),
                     "prophet_bstar_relmae": round(b, 4) if np.isfinite(b) else np.nan,
                     "delta_relmae_vs_bstar": round(b - best["relmae_vs_snaive"], 4) if np.isfinite(b) else np.nan,
                     "ratio_vs_bstar": round(best["relmae_vs_snaive"] / b, 4) if np.isfinite(b) else np.nan,
                     "n_series": int(best["n_series"]), "n_origins": int(best["n_origins"])})
    out = pd.DataFrame(rows).sort_values(["category", "horizon"])
    out.to_csv(ROOT / "outputs" / "T5_category_champions.csv", index=False)
    print(out.to_string(index=False), flush=True)

    # --- Маркетплейсы: топ-5 по горизонтам + тренд-гипотеза ---
    mp = t2[t2.category == "Маркетплейсы"]
    print("\n=== Маркетплейсы: топ-5 по RelMAE на горизонте ===", flush=True)
    for h in HORIZONS:
        top = mp[mp.horizon == h].nsmallest(5, "relmae_vs_snaive")
        print(f"h={h}: " + " | ".join(f"{r.model} {r.relmae_vs_snaive:.3f}" for r in top.itertuples()), flush=True)

    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    wide = panel.pivot_table(index="series_id", columns="month_idx", values="log_value").reindex(columns=range(24))
    cats = wide.index.str.split("|").str[1]
    # pooled median YoY на train-окне (месяц 12 vs 0, 13 vs 1, ..., 23 vs 11 -> все <= 23,
    # но для ex-ante берём окно origins: YoY по месяцам 12..23 против 0..11)
    yoy = wide.iloc[:, 12:].to_numpy() - wide.iloc[:, :12].to_numpy()  # 12 пар
    print("\n=== Тренд (pooled median YoY log-прирост, %/год) по категориям ===", flush=True)
    for c in sorted(cats.unique()):
        m = np.nanmedian(yoy[cats == c]) * 100
        print(f"  {c}: {m:+.1f}%", flush=True)
    # волатильность после детренда нет нужды — точка ровно в уровне тренда


if __name__ == "__main__":
    main()
