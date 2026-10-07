"""rc1 / O3 бонус: квантильный бленд Chronos-2 (q10/q90) × conformal-полосы чемпиона.

Общий origin=12 (единственный, где есть квантили Chronos по всем шагам 1..12).
Бленд = среднее границ: lo=(q10+lo80)/2, hi=(q90+hi80)/2.
Метрики на matched cells (series × horizon, где есть обе полосы): coverage80 и
средняя ширина (в рублях и в % от y_true) для Chronos / conformal / blend.

Запуск: .venv/bin/python scripts/rc1_o3_quantile_blend.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    q = pd.read_parquet(ROOT / "outputs" / "preds_chronos2_quant_o12.parquet")
    q = q[q["origin"] == 12][["series_id", "horizon", "y_true", "q10", "q90"]]
    iv = pd.read_parquet(ROOT / "outputs" / "intervals_champion.parquet")
    iv = iv[(iv["origin"] == 12) & (iv["model"] == "DriftAdjSNaive")][
        ["series_id", "horizon", "lo80", "hi80"]]
    m = q.merge(iv, on=["series_id", "horizon"], how="inner")
    print(f"matched cells (origin 12, DriftAdjSNaive × Chronos quant): {len(m)}")

    m["blend_lo"] = (m["q10"] + m["lo80"]) / 2
    m["blend_hi"] = (m["q90"] + m["hi80"]) / 2

    rows = []
    for h, g in m.groupby("horizon"):
        for name, lo, hi in [("chronos_q10q90", g["q10"], g["q90"]),
                             ("conformal_DriftAdj", g["lo80"], g["hi80"]),
                             ("blend_mean_bounds", g["blend_lo"], g["blend_hi"])]:
            cov = ((g["y_true"] >= lo) & (g["y_true"] <= hi)).mean()
            width = (hi - lo).mean()
            width_pct = ((hi - lo) / g["y_true"]).mean() * 100
            rows.append({"horizon": h, "interval": name, "coverage80": cov,
                         "width_rub": width, "width_pct_of_y": width_pct,
                         "n_cells": len(g)})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "rc1_o3_quantile_blend.csv", index=False)
    piv = res.pivot_table(index="horizon", columns="interval",
                          values=["coverage80", "width_rub"])
    pd.set_option("display.width", 200)
    print(piv.round(3).to_string(), flush=True)
    # сводка «номинал 80%»
    print("\ncoverage по всем шагам (mean): ",
          res.groupby("interval")["coverage80"].mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
