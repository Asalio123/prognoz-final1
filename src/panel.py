"""Построение чистой панели МО из сырых данных СберИндекса (CC BY-SA 4.0).

Сырые parquet НЕ коммитятся (п.9.2.2–9.2.3 Положения); panel.parquet — производная, тоже вне git.
Решения зафиксированы в configs/validation.yaml: только полные 24-мес ряды, фильтр медианы <300 руб.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "panel"
OUT.mkdir(parents=True, exist_ok=True)

MONTHS = pd.period_range("2023-01", "2024-12", freq="M").astype(str).tolist()


def build_panel() -> pd.DataFrame:
    cons = pd.read_parquet(RAW / "hackathon" / "hackathonlicence" / "consumption.parquet")
    cons["value"] = cons["value"].astype("float64")

    # только полные 24-месячные ряды (ядро 2016 МО; 174 неполных — региональные окна/новые МО, E3)
    sizes = cons.groupby(["territory_id", "category"])["date"].nunique()
    complete = sizes[sizes == 24].index
    cons = cons.set_index(["territory_id", "category"]).loc[complete].reset_index()

    # фильтр шумовых микро-рядов: медиана < 300 руб/мес (E3: малые МО, категория Общепит)
    med = cons.groupby(["territory_id", "category"])["value"].median()
    noisy = med[med < 300].index
    cons = cons[~cons.set_index(["territory_id", "category"]).index.isin(noisy)]

    cons["log_value"] = np.log1p(cons["value"])
    cons["month_idx"] = cons["date"].map({m: i for i, m in enumerate(MONTHS)})

    # справочник: имя/регион/тип/координаты для интерпретации и разрезов
    d = pd.read_excel(RAW / "t_dict_municipal_districts.xlsx")
    d = d[d["year_to"] >= 2023]  # действующие в окне панели
    d = d.sort_values("year_from").drop_duplicates("territory_id", keep="last")
    meta = d[["territory_id", "municipal_district_name_short", "municipal_district_type",
              "municipal_district_status", "region_name", "oktmo",
              "municipal_district_center_lat", "municipal_district_center_lon"]]
    cons = cons.merge(meta, on="territory_id", how="left")

    cons.to_parquet(OUT / "panel.parquet")
    return cons


if __name__ == "__main__":
    p = build_panel()
    n_mo = p["territory_id"].nunique()
    print(f"panel: {len(p)} строк, {n_mo} МО, категории: {sorted(p['category'].unique())}")
    print(p.groupby("category")["territory_id"].nunique())
