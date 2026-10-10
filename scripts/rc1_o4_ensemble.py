"""Ансамбли CPD-детекторов на matched-бюджете (К3).

Пересчитывает matched-логику scripts/26_matched_rate.py (те же сиды 42/123, тот же
вход — десезонированный лог-ряд из src.cpd, те же скореры — импортируются из 26,
ничего не переопределяется) для методов {PELT, Window, BOCPD} и строит:

1. Верификацию: пересчитанные строки PELT/Window/BOCPD обязаны побитово совпасть
   с outputs/cpd_matched_rate_v3.csv (иначе — стоп, мой пайплайн нефиdelен).
2. Голосования на инъекциях (посерийные топ-K пики каждого метода):
   - vote2of3_pm1: брейк там, где >=2 методов из {BOCPD, Window, PELT} ставят брейк
     в окне +/-1 мес (кластеризация позиций, центр = округлённое среднее);
   - bocpd_or_window: объединение пиков BOCPD и Window (дедупликация +/-1 мес);
   - ens2of3_topK: бюджетно-совместимый вариант — кластеры голосов ранжируются по
     (число голосов, сумма min-max-нормированных скоров сторонников), отбираются
     топ-K_panel с NMS +/-2 (тот же NMS, что в 26) — строка, сравнимая с фронтиром.
   После каждого правила — NMS +/-2 внутри ряда (как в 26), оценка eval_detection
   с tol=2 (как в 26).
3. Реальная панель: голосование на outputs/cpd_breaks_v2.parquet (замороженные
   рабочие точки), счётчики брейков и FAR/1000 МО-мес.

CPD-стенд (scripts/06*, калибровка/фронтир) НЕ перегоняется; пересчёт только
matched-скореров 26 — разрешён брифом явно. Выходы:
  outputs/rc1_o4_ensemble_votes.csv    — голосования с естественным бюджетом
  outputs/rc1_o4_ensemble_matched.csv  — бюджетно-совместимый ансамбль vs BOCPD
  outputs/rc1_o4_realpanel_votes.csv   — голосования на реальной панели (счётчики)
"""
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.cpd import (category_profiles, deseasonalize, mad_scale, inject,  # noqa: E402
                     eval_detection, load_wide, stratified_measurement_idx,
                     injection_plans, CELLS, K_SHRINK, DETREND, ROBUST)

m26 = import_module("26_matched_rate")

METHODS = ["PELT", "Window", "BOCPD"]
SCORERS = {m: m26.SCORERS[m] for m in METHODS}
POS_LO, POS_HI, NMS_TOL, EVAL_TOL = m26.POS_LO, m26.POS_HI, m26.NMS_TOL, 2
VOTE_TOL = 1  # окно голосования "+/-1 мес" из брифа


def cluster_positions(items, tol=VOTE_TOL):
    """items: [(pos, method, norm_score)] одного ряда -> кластеры позиций в окне ±tol.

    Жадная кластеризация по отсортированным позициям: новый элемент входит в кластер,
    если ближе чем tol+1 к последнему члену кластера (chain merge).
    Возвращает [{pos_center, votes, methods, score_sum}].
    """
    items = sorted(items, key=lambda x: x[0])
    clusters, cur = [], []
    for it in items:
        if cur and it[0] - cur[-1][0] <= tol:
            cur.append(it)
        else:
            if cur:
                clusters.append(cur)
            cur = [it]
    if cur:
        clusters.append(cur)
    out = []
    for c in clusters:
        methods = {m for _, m, _ in c}
        out.append(dict(pos=int(round(np.mean([p for p, _, _ in c]))),
                        votes=len(methods),
                        methods=methods,
                        score_sum=sum(s for _, _, s in c)))
    return out


def nms_positions(cands, k=None, nms_tol=NMS_TOL):
    """cands: [(sort_key_desc, pos)] -> отобранные позиции с NMS внутри ряда."""
    picked = []
    for _, pos in sorted(cands, key=lambda c: -c[0]):
        if any(abs(pos - p) <= nms_tol for p in picked):
            continue
        picked.append(pos)
        if k is not None and len(picked) >= k:
            break
    return picked


def main():
    t0 = time.time()
    wide, cats = load_wide(ROOT)
    W = wide.to_numpy()
    profs = category_profiles(W, cats)
    prof_arr = np.array([profs[c] for c in cats])

    def ds(y, i):
        return deseasonalize(y, prof_arr[i], k=K_SHRINK, detrend=DETREND, robust=ROBUST)

    meas = stratified_measurement_idx(W, cats)
    plans = injection_plans(len(meas))
    n = len(meas)
    print(f"измерительных рядов: {n}; ячеек: {len(CELLS)}", flush=True)

    ref = pd.read_csv(ROOT / "outputs" / "cpd_matched_rate_v3.csv")
    vote_rows, matched_rows, verify_rows = [], [], []

    for kind, d in CELLS:
        taus = plans[(kind, d)]
        # --- z-ряды ячейки и бюджет K (идентично 26)
        zs_cell, k_cell = [], 0
        for j, i in enumerate(meas):
            y = W[i]
            z = ds(inject(y, kind, int(taus[j]), d, s=mad_scale(y)), i)
            zs_cell.append(z)
            k_cell += len(m26.breaks_pelt_bic(z))
        # --- кандидаты трёх методов (идентично 26)
        per_method = {m: [] for m in METHODS}
        for j, z in enumerate(zs_cell):
            for m, fn in SCORERS.items():
                for pos, sc in fn(z).items():
                    if POS_LO <= pos < POS_HI and np.isfinite(sc):
                        per_method[m].append((sc, j, pos))
        # --- min-max нормировка скоров метода по панели ячейки (для тай-брейка)
        norm = {}
        for m in METHODS:
            arr = np.array([c[0] for c in per_method[m]])
            lo, hi = arr.min(), arr.max()
            norm[m] = (lambda sc, lo=lo, hi=hi: (sc - lo) / (hi - lo) if hi > lo else 0.0)
        # --- топ-K пики каждого метода (идентично 26)
        picked = {m: m26.select_top_k(per_method[m], k_cell) for m in METHODS}

        # --- верификация против сохранённой таблицы
        for m in METHODS:
            tp = fp = fn = 0
            for j in range(n):
                r = eval_detection(picked[m].get(j, []), int(taus[j]), tol=EVAL_TOL)
                tp += r[0]; fp += r[1]; fn += r[2]
            row = ref[(ref.method == m) & (ref.kind == kind) & (ref.delta == d)].iloc[0]
            ok = (tp == row.tp) and (fp == row.fp) and (fn == row.fn) and (k_cell == row.K_panel)
            verify_rows.append(dict(kind=kind, delta=d, method=m, tp=tp, fp=fp, fn=fn,
                                    K=k_cell, ref_tp=row.tp, ref_fp=row.fp, ref_fn=row.fn,
                                    match=bool(ok)))

        def score_set(pos_list_per_series):
            tp = fp = fn = 0
            delays = []
            nsel = 0
            for j in range(n):
                found = pos_list_per_series.get(j, [])
                nsel += len(found)
                r = eval_detection(found, int(taus[j]), tol=EVAL_TOL)
                tp += r[0]; fp += r[1]; fn += r[2]
                if not np.isnan(r[3]):
                    delays.append(r[3])
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            return dict(n_selected=nsel, tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
                        F1=2 * prec * rec / max(prec + rec, 1e-9),
                        delay_mean=float(np.mean(delays)) if delays else np.nan)

        # --- правило 1: >=2 из 3 в окне ±1 (естественный бюджет)
        v23 = {}
        for j in range(n):
            items = []
            for m in METHODS:
                for pos in picked[m].get(j, []):
                    sc = max((c[0] for c in per_method[m] if c[1] == j and c[2] == pos),
                             default=0.0)
                    items.append((pos, m, norm[m](sc)))
            cls = [c for c in cluster_positions(items) if c["votes"] >= 2]
            v23[j] = nms_positions([(c["votes"] + c["score_sum"] / 10.0, c["pos"]) for c in cls])
        r = score_set(v23)
        vote_rows.append(dict(rule="vote2of3_pm1", kind=kind, delta=d, K_panel=k_cell, **r))

        # --- правило 2: BOCPD ∨ Window (естественный бюджет)
        bow = {}
        for j in range(n):
            items = ([(p, "BOCPD", 0.0) for p in picked["BOCPD"].get(j, [])] +
                     [(p, "Window", 0.0) for p in picked["Window"].get(j, [])])
            cls = cluster_positions(items)
            bow[j] = nms_positions([(1.0, c["pos"]) for c in cls])
        r = score_set(bow)
        vote_rows.append(dict(rule="bocpd_or_window", kind=kind, delta=d, K_panel=k_cell, **r))

        # --- правило 3: бюджетно-совместимый ансамбль (панельный топ-K кластеров голосов)
        allc = []
        for j in range(n):
            items = []
            for m in METHODS:
                for pos in picked[m].get(j, []):
                    sc = max((c[0] for c in per_method[m] if c[1] == j and c[2] == pos),
                             default=0.0)
                    items.append((pos, m, norm[m](sc)))
            for c in cluster_positions(items):
                allc.append((c["votes"] + c["score_sum"] / 10.0, j, c["pos"]))
        ensK = m26.select_top_k(allc, k_cell)
        r = score_set(ensK)
        matched_rows.append(dict(rule="ens_votes_topK", kind=kind, delta=d, K_panel=k_cell, **r))
        r = score_set(picked["BOCPD"])
        matched_rows.append(dict(rule="BOCPD_single", kind=kind, delta=d, K_panel=k_cell, **r))
        r = score_set(picked["Window"])
        matched_rows.append(dict(rule="Window_single", kind=kind, delta=d, K_panel=k_cell, **r))
        r = score_set(picked["PELT"])
        matched_rows.append(dict(rule="PELT_single", kind=kind, delta=d, K_panel=k_cell, **r))
        print(f"ячейка {kind} {d}: K={k_cell}, {time.time()-t0:.0f}с", flush=True)

    ver = pd.DataFrame(verify_rows)
    v = pd.DataFrame(vote_rows)
    mt = pd.DataFrame(matched_rows)

    v.to_csv(ROOT / "outputs" / "rc1_o4_ensemble_votes.csv", index=False)
    mt.to_csv(ROOT / "outputs" / "rc1_o4_ensemble_matched.csv", index=False)

    print("\n=== ВЕРИФИКАЦИЯ против cpd_matched_rate_v3.csv ===")
    print(f"совпало строк: {ver['match'].sum()}/{len(ver)}")
    if not ver["match"].all():
        print(ver[~ver["match"]].to_string(index=False))

    print("\n=== Голосования (естественный бюджет), level-ячейки ===")
    print(v[v.kind == "level"].round(3).to_string(index=False))
    print("\n=== Бюджетно-совместимый ансамбль vs одиночки, level-ячейки ===")
    print(mt[mt.kind == "level"].round(3).to_string(index=False))

    # --- реальная панель: голосование на замороженных брейках
    br = pd.read_parquet(ROOT / "outputs" / "cpd_breaks_v2.parquet")
    n_series_total = br[["territory_id", "category"]].drop_duplicates().shape[0]
    # месяцев в панели = 24 (1..24 точки; брейки в 1..23)
    denom = len(pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
                [["territory_id", "category"]].drop_duplicates()) * 24
    real_rows = []
    grp = br.groupby(["territory_id", "category"])
    picks_real = {m: {} for m in METHODS}
    for (tid, cat), g in grp:
        for m, gm in g.groupby("method"):
            if m in picks_real:
                picks_real[m][(tid, cat)] = sorted(gm["break_month"].astype(int).tolist())
    for m in METHODS:
        cnt = sum(len(vv) for vv in picks_real[m].values())
        real_rows.append(dict(rule=f"{m}_single", n_breaks=cnt,
                              far_per_1000=cnt / denom * 1000))
    v23r, bowr = {}, {}
    keys = set()
    for m in METHODS:
        keys |= set(picks_real[m].keys())
    for k in keys:
        items = [(p, m, 0.0) for m in METHODS for p in picks_real[m].get(k, [])]
        cls = cluster_positions(items)
        v23r[k] = nms_positions([(c["votes"], c["pos"]) for c in cls if c["votes"] >= 2])
        items2 = [(p, m, 0.0) for m in ("BOCPD", "Window") for p in picks_real[m].get(k, [])]
        bowr[k] = nms_positions([(1.0, c["pos"]) for c in cluster_positions(items2)])
    for name, dd in [("vote2of3_pm1", v23r), ("bocpd_or_window", bowr)]:
        cnt = sum(len(vv) for vv in dd.values())
        real_rows.append(dict(rule=name, n_breaks=cnt, far_per_1000=cnt / denom * 1000))
    rp = pd.DataFrame(real_rows)
    rp.to_csv(ROOT / "outputs" / "rc1_o4_realpanel_votes.csv", index=False)
    print("\n=== Реальная панель (замороженные рабочие точки), счётчики ===")
    print(rp.round(4).to_string(index=False))
    print(f"\ntotal {time.time()-t0:.0f}с", flush=True)


if __name__ == "__main__":
    main()
