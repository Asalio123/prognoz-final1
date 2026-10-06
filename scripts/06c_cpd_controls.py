"""CPD контрольные проверки v2 (К3): позитивные кейсы, негативные контроли, стабильность.

Закрывает из аудита A11/A05:
- негативные контроли R04 §2.2: якутско-тывинские декабри, ноябрьские распродажи,
  январская яма-2024 — FP-rate на замороженных рабочих точках (в rc0: Якутия 14.9%,
  Тыва 25.0% — провал; январь-2024 размечался у 62.6% рядов);
- матрица В (позитивные кейсы E3/C3) по замороженному скану cpd_breaks_v2.parquet;
- гейты стабильности R04 §3 для финалиста фронтира: Jaccard block-bootstrap
  (блок 6 мес, толерантность ±1) и extension flip-rate (t=18..23).

Выход: outputs/cpd_controls_v2.csv, outputs/cpd_matrix_B_v2.csv,
outputs/cpd_stability_v2.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cpd import (category_profiles, deseasonalize, bocpd_prestart, load_wide,  # noqa: E402
                     stratified_measurement_idx, detect_all, working_points,
                     GRIDS, FAR_BUDGET, K_SHRINK, DETREND, ROBUST)

# позитивные кейсы (E3 §6, C3 GT): (название, tid, категория, истинный idx брейка)
CASES = [
    ("Богородский", 1267, "Маркетплейсы", 13),          # 2024-02, +402% level
    ("Дальнереченский", 788, "Общественное питание", 5),  # 2023-06, +504% level
]
NATIONAL = [
    # (название, категория, idx, минимальная доля МО для «поймано»)
    ("Яйца-2023.12", "Продовольствие", 11, 0.50),
    ("Ипотека-2024.07", "Все категории", 18, 0.50),
    ("Девальвация-2024.11", "Все категории", 22, 0.50),
]
FLOOD_TIDS = {"Орск": None}  # заполняется по имени МО ниже


def block_bootstrap_resid(z, block, rng):
    """Circular block-bootstrap остатков (R04 §3: блок 6 мес)."""
    n = len(z)
    out = []
    while sum(map(len, out)) < n:
        start = rng.integers(0, n)
        b = np.array([z[(start + k) % n] for k in range(block)])
        out.append(b)
    return np.concatenate(out)[:n]


def match_count(a, b, tol=1):
    """|A∩B| с толерантностью tol (жадно по ближайшим)."""
    a, b = sorted(a), sorted(b)
    used = [False] * len(b)
    m = 0
    for x in a:
        for j, yv in enumerate(b):
            if not used[j] and abs(x - yv) <= tol:
                used[j] = True
                m += 1
                break
    return m


def jaccard(a, b, tol=1):
    m = match_count(a, b, tol)
    denom = len(a) + len(b) - m
    return m / denom if denom else 1.0


def main():
    t0 = time.time()
    wide, cats = load_wide(ROOT)
    W = wide.to_numpy()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    meta = panel.drop_duplicates("series_id").set_index("series_id")
    profs = category_profiles(W, cats)
    prof_arr = np.array([profs[c] for c in cats])
    ds = lambda y, i: deseasonalize(y, prof_arr[i], k=K_SHRINK, detrend=DETREND, robust=ROBUST)
    sids = wide.index.to_numpy()
    tids = np.array([int(s.split("|")[0]) for s in sids])

    cal = pd.read_csv(ROOT / "outputs" / "cpd_calibration_v2.csv")
    frozen = working_points(cal, FAR_BUDGET)
    print("замороженные точки:", frozen, flush=True)
    br = pd.read_parquet(ROOT / "outputs" / "cpd_breaks_v2.parquet")
    br["sid"] = br.territory_id.astype(str) + "|" + br.category

    # --- негативные контроли (FP-rate по рядам, по каждому методу на рабочей точке)
    reg = meta["region_name"]
    neg_groups = {
        "Якутия": reg.str.contains("Саха").to_numpy(),
        "Тыва": reg.str.contains("Тыва").to_numpy(),
    }
    sid2pos = {s: i for i, s in enumerate(sids)}
    rows = []
    for m, p in frozen.items():
        if p is None:
            continue
        sub = br[br.method == m]
        per_sid = sub.groupby("sid")["break_month"].apply(set)
        # якутско-тывинские декабри: брейк в 11 или 23 (граница перед декабрём)
        for gname, gmask in neg_groups.items():
            gids = set(reg[gmask].index)
            hits = sum(1 for s in gids if s in per_sid.index
                       and ({11, 23} & set(per_sid[s])))
            n_g = len(gids)
            rows.append(dict(method=m, control=f"{gname}-декабрь", fp_rate=hits / max(n_g, 1), n=n_g))
        # ноябрьские распродажи: брейк в 10 или 22 по всей панели
        hits = sum(1 for s in sids if s in per_sid.index and ({10, 22} & set(per_sid[s])))
        rows.append(dict(method=m, control="ноябрь-распродажи", fp_rate=hits / len(sids), n=len(sids)))
        # январская яма-2024: брейк ровно на границе 12
        hits = sum(1 for s in sids if s in per_sid.index and (12 in per_sid[s]))
        rows.append(dict(method=m, control="январская-яма-2024", fp_rate=hits / len(sids), n=len(sids)))
        # доля брейков в одном месяце (концентрация-артефакт)
        vc = sub.break_month.value_counts(normalize=True)
        rows.append(dict(method=m, control="макс.доля_месяца", fp_rate=vc.iloc[0] if len(vc) else 0.0,
                         n=int(vc.index[0]) if len(vc) else -1))
    neg = pd.DataFrame(rows)
    neg.to_csv(ROOT / "outputs" / "cpd_controls_v2.csv", index=False)
    print(neg.round(4).to_string(index=False), flush=True)

    # --- матрица В: позитивные кейсы (hit@±2)
    mrows = []
    for m, p in frozen.items():
        sub = br[br.method == m]
        row = {"method": m}
        for cname, tid, cat, tau in CASES:
            sid = f"{tid}|{cat}"
            got = sub[sub.sid == sid].break_month.tolist()
            hit = any(abs(b - tau) <= 2 for b in got)
            row[cname] = f"{'HIT' if hit else 'MISS'} {got if got else '-'}"
        for cname, cat, tau, thr_share in NATIONAL:
            ss = sub[(sub.category == cat)]
            n_mo = ss.territory_id.nunique()
            hit_mo = ss[(ss.break_month >= tau - 2) & (ss.break_month <= tau + 2)].territory_id.nunique()
            share = hit_mo / max(n_mo, 1)
            row[cname] = f"{share*100:.0f}% ({'PASS' if share >= thr_share else 'FAIL'})"
        mrows.append(row)
    mb = pd.DataFrame(mrows)
    mb.to_csv(ROOT / "outputs" / "cpd_matrix_B_v2.csv", index=False)
    print(mb.to_string(index=False), flush=True)

    # --- стабильность для финалистов (Window + PELT): Jaccard + flip-rate
    meas = stratified_measurement_idx(W, cats)
    rng = np.random.default_rng(5)
    stab_rows = []
    for m in ("Window", "PELT"):
        p = frozen.get(m)
        if p is None:
            continue
        # Jaccard block-bootstrap на остатках (конфиг детектора тот же)
        jacs = []
        for i in meas[:150]:
            z = ds(W[i], i)
            base = detect_all(z, bocpd_prestart(z), m, p)
            for _ in range(20):
                zb = block_bootstrap_resid(z, 6, rng)
                jb = detect_all(zb, bocpd_prestart(zb), m, p)
                jacs.append(jaccard(base, jb, tol=1))
        # flip-rate: брейки на z[:t] vs z[:t+1], t = 18..22
        flips, tot = 0, 0
        for i in meas:
            y = W[i]
            prev = None
            for t in range(18, 24):
                zt = ds(y[:t], i)
                cur = set(detect_all(zt, bocpd_prestart(zt), m, p))
                if prev is not None:
                    for b in prev:
                        tot += 1
                        if not any(abs(b - c) <= 1 for c in cur):
                            flips += 1
                prev = cur
        stab_rows.append(dict(method=m, param=p,
                              jaccard_mean=np.mean(jacs), jaccard_p25=np.percentile(jacs, 25),
                              flip_rate=flips / max(tot, 1), n_flip_base=tot))
        print(f"стабильность {m}: {time.time()-t0:.0f}с", flush=True)
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(ROOT / "outputs" / "cpd_stability_v2.csv", index=False)
    print(stab.round(3).to_string(index=False), flush=True)
    print(f"total {time.time()-t0:.0f}с", flush=True)


if __name__ == "__main__":
    main()
