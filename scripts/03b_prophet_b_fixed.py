"""Prophet-B-fixed: честная конфигурация B = правило, а не прогон.

Находка аудита rc0 (A09, верифицировано): катастрофа Prophet-B целиком локализована
в origin=12, где train-окно = 334 дня < 365.25 — годовая сезонность с 4 парами Фурье
недоопределена (Prophet сам предупреждает об under-identification при <2 циклах).
На origins 13..23 B систематически сильнее A.

Честный B (фиксируем как Prophet-B* в отчёте): «yearly=4 включается только когда
train-окно покрывает полный годовой цикл» — т.е. origin 12 → конфиг A, origins ≥13 → B.
Предсказания обеих ветвей уже посчитаны (03_run_prophet.py), поэтому splice точный,
без нового compute.

Запуск: python scripts/03b_prophet_b_fixed.py
Выход: outputs/preds_prophet_Bfixed.parquet
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

pa = pd.read_parquet(ROOT / "outputs" / "preds_prophet_A.parquet")
pb = pd.read_parquet(ROOT / "outputs" / "preds_prophet_B.parquet")

a12 = pa[pa["origin"] == 12][["series_id", "origin", "horizon", "y_pred"]].rename(
    columns={"y_pred": "y_pred_a"})
bf = pb.merge(a12, on=["series_id", "origin", "horizon"], how="left", validate="1:1")
n_swapped = int((bf["origin"] == 12).sum())
bf["y_pred"] = bf["y_pred_a"].where(bf["origin"] == 12, bf["y_pred"])
bf = bf.drop(columns=["y_pred_a"])
bf["model"] = "Prophet-B*"

out = ROOT / "outputs" / "preds_prophet_Bfixed.parquet"
bf.to_parquet(out, index=False)
print(f"saved {out}: {len(bf)} строк, подменено origin-12 ячеек: {n_swapped}")
