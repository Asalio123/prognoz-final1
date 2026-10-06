"""Проверка пространственного распространения шоков (заменяет число-призрак «0/25856»,
снятое аудитом rc0: скрипта не существовало, а ноль противоречил именному кейсу).

Дизайн (честный, посчитанный на данных):
- события = детектированные брейки (outputs/cpd_breaks_v2.parquet, метод Window — рабочая
  точка фронтира; sensitivity: BOCPD);
- пары = соседи по автодорожному расстоянию ≤ 80 км (connection.parquet);
- «распространение» = брейк в МО i в месяце t И брейк в МО j в месяце t+1..t+4,
  та же категория;
- нулевая модель = circular-shift месяцев брейков (сохраняет автокорреляцию), 1000 повторов.
Выход: outputs/spatial_propagation.csv + консольный вердикт.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAX_KM = 80.0
WINDOW = 4


def count_linked(breaks: pd.DataFrame, pairs: pd.DataFrame) -> int:
    """Число пар (i→j), где j имеет брейк той же категории через 1..WINDOW мес после i."""
    ev_i = breaks[["territory_id", "category", "break_month"]].dropna().rename(
        columns={"territory_id": "territory_id_x", "break_month": "bm_i"})
    ev_i["bm_i"] = ev_i["bm_i"].astype(int)
    ev_j = breaks[["territory_id", "category", "break_month"]].dropna().rename(
        columns={"territory_id": "territory_id_y", "break_month": "bm_j"})
    ev_j["bm_j"] = ev_j["bm_j"].astype(int)
    m = pairs.merge(ev_i, on="territory_id_x")
    m = m.merge(ev_j, on=["territory_id_y", "category"])
    lag = m["bm_j"] - m["bm_i"]
    return int(((lag >= 1) & (lag <= WINDOW)).sum())


def main():
    pairs = pd.read_parquet(ROOT / "data" / "raw" / "hackathon" / "hackathonlicence" / "connection.parquet")
    pairs = pairs[pairs["distance"] <= MAX_KM]
    for c in ("territory_id_x", "territory_id_y"):
        pairs[c] = pairs[c].astype(int)
    print(f"пар ≤{MAX_KM} км: {len(pairs)}", flush=True)

    br = pd.read_parquet(ROOT / "outputs" / "cpd_breaks_v2.parquet")
    br["territory_id"] = br["territory_id"].astype(int)
    rng = np.random.default_rng(42)
    rows = []
    for method in sorted(br["method"].unique()):
        b = br[br["method"] == method]
        obs = count_linked(b, pairs)
        # нулевая модель: circular shift месяцев брейков внутри ряда-категории
        nulls = []
        for rep in range(1000):
            shift = rng.integers(1, 24)
            bs = b.copy()
            bs["break_month"] = (bs["break_month"] + shift) % 24
            nulls.append(count_linked(bs, pairs))
        nulls = np.array(nulls)
        p = (nulls >= obs).mean()
        lift = obs / max(nulls.mean(), 1e-9)
        rows.append({"method": method, "pairs_le80km": len(pairs), "breaks": len(b),
                     "observed_linked": obs, "null_mean": round(nulls.mean(), 2),
                     "lift": round(lift, 3), "p_value": round(p, 4)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "spatial_propagation.csv", index=False)
    print("saved outputs/spatial_propagation.csv", flush=True)


if __name__ == "__main__":
    main()
