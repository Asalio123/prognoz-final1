"""23b. Ручная проверка конформального пайплайна на 2 симулированных рядах.

Двойной счёт:
(а) векторный путь — те же функции, что в 23_champion_intervals.py;
(б) наивный путь — чистый Python (math.log1p, sorted, индексация), без numpy,
    каждое промежуточное число печатается и перевычисляется калькулятором.
Плюс литеральные assert'ы на заранее известные значения (квантиль, пулы).

Запуск: .venv/bin/python scripts/23b_intervals_selftest.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
iv = import_module("23_champion_intervals") if False else None
# имя модуля начинается с цифры — грузим через importlib.util
import importlib.util
spec = importlib.util.spec_from_file_location(
    "champ_iv", Path(__file__).resolve().parent / "23_champion_intervals.py")
champ_iv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(champ_iv)

# ---------------------------------------------------------- игрушечные ряды
# A: 1000 + 10*t, декабри (t=11,23) x1.5.  B: 500 ровно, декабри x2.
A = [1000 + 10 * t for t in range(24)]
A[11], A[23] = int(1110 * 1.5), int(1230 * 1.5)
B = [500] * 24
B[11], B[23] = 1000, 1000
RUB = [A, B]
WV_L = [[math.log1p(v) for v in row] for row in RUB]   # list-форма для наивного пути
wv = np.array(WV_L)                                     # numpy-форма для векторного
REG = ["R1", "R2"]


# ---------------------------------------------------------- наивные модели
def naive_drift(t, h):
    """DriftAdjSNaive as-of, чистый Python. Возвращает [predA, predB] в log."""
    c = t - h
    if c < 0:
        return None
    med = [sorted(WV_L[s][:c + 1])[len(WV_L[s][:c + 1]) // 2]
           if len(WV_L[s][:c + 1]) % 2 == 1 else
           sum(sorted(WV_L[s][:c + 1])[len(WV_L[s][:c + 1]) // 2 - 1:
                                      len(WV_L[s][:c + 1]) // 2 + 1]) / 2
           for s in (0, 1)]
    g = 0.0
    if c >= 12:
        yoy = sorted(WV_L[s][c] - WV_L[s][c - 12] for s in (0, 1))
        g = sum(yoy) / 2  # медиана двух = среднее
    out = []
    for s in (0, 1):
        out.append(WV_L[s][t - 12] + g if t >= 12 else med[s] + g)
    return out


def naive_seas(t, h):
    """SeasProfileHA as-of, чистый Python (k=2, региональный шринкаж)."""
    c = t - h
    if c < 0:
        return None
    train = list(range(c + 1))
    mcols = [i for i in train if i % 12 == t % 12]
    out = []
    med_all = []
    for s in (0, 1):
        vals = sorted(WV_L[s][i] for i in train)
        n = len(vals)
        med_all.append(vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2)
    if mcols:
        # региональный профиль: в каждом регионе один ряд -> ratio региона =
        # med(mcols) - med(train) самого ряда; mo_seas то же самое =>
        # seas = (m*mo + 2*reg)/(m+2) = mo (при 1 ряде на регион). Пишем честно.
        reg_ratio = []
        for s in (0, 1):
            mv = sorted(WV_L[s][i] for i in mcols)
            mm = mv[len(mv) // 2] if len(mv) % 2 else (mv[len(mv) // 2 - 1] + mv[len(mv) // 2]) / 2
            reg_ratio.append(mm - med_all[s])
        for s in (0, 1):
            mo_seas = reg_ratio[s]  # тот же med(mcols) - med_all
            seas = (len(mcols) * mo_seas + 2.0 * reg_ratio[s]) / (len(mcols) + 2.0)
            out.append(med_all[s] + seas)
    else:
        out = [med_all[0], med_all[1]]
    return out


# ---------------------------------------------------------- 1) модели совпадают
print("=== 1) as-of модели: вектор vs наивный ===")
worst = 0.0
for h in (1, 3, 6, 12):
    for t in range(h, 24):
        vn = champ_iv.drift_pred_log(wv, t, h)
        nn = naive_drift(t, h)
        if nn is None:
            assert vn is None
            continue
        d = max(abs(vn[s] - nn[s]) for s in (0, 1))
        worst = max(worst, d)
        vs = champ_iv.seasprofile_pred_log(wv, np.array(REG), t, h)
        ns = naive_seas(t, h)
        d2 = max(abs(vs[s] - ns[s]) for s in (0, 1))
        worst = max(worst, d2)
assert worst < 1e-12, worst
print(f"DriftAdjSNaive и SeasProfileHA: max |вектор-наивный| = {worst:.2e}  OK")

# ---------------------------------------------------------- 2) conf_quant литерально
print("=== 2) conf_quant на известном списке ===")
scores = [0.01 * i for i in range(1, 11)]  # 0.01..0.10, n=10
q80, n = champ_iv.conf_quant(scores, 0.20)
# k = ceil(11*0.8) = 9 -> 9-й наименьший = 0.09
assert n == 10 and abs(q80 - 0.09) < 1e-12, (q80, n)
q95, _ = champ_iv.conf_quant(scores, 0.05)
# k = ceil(11*0.95) = 11 -> обрезка до n=10 -> 0.10
assert abs(q95 - 0.10) < 1e-12, q95
print(f"n=10: q80 = k9 = 0.09 (получено {q80}); q95 = k11->clip10 = 0.10 "
      f"(получено {q95})  OK")

# ---------------------------------------------------------- 3) select_pool литерально
print("=== 3) select_pool: границы tier ===")
ts, hs, tier = champ_iv.select_pool(15, 1)
assert ts == [12, 13, 14] and tier == 1, (ts, tier)
ts, hs, tier = champ_iv.select_pool(12, 1)
assert ts == list(range(1, 12)) and tier == 2, (ts, tier)
ts, hs, tier = champ_iv.select_pool(12, 12)
assert tier == 3 and len(ts) == 26, (tier, len(ts))  # 11+9+6+0 кросс-h пар
print("o=15,h=1 -> T1 t=[12,13,14]; o=12,h=1 -> T2 t=[1..11]; "
      "o=12,h=12 -> T3 26 пар  OK")

# ---------------------------------------------------------- 4) интервал и покрытие руками
print("=== 4) интервал (o=14, h=1) и покрытие — двойной счёт ===")
o, h = 14, 1
tm = o + h - 1  # 14
# (а) векторный путь
res = champ_iv.build_residuals(wv, np.array(REG))[0]["DriftAdjSNaive"][h]
ts, hs_, tier = champ_iv.select_pool(o, h)
pool_vec = np.concatenate([res[:, t] for t in ts])
q80v, _ = champ_iv.conf_quant(pool_vec, 0.20)
pred_v = champ_iv.drift_pred_log(wv, tm, h)
# (б) наивный путь: остатки t=12,13 обоих рядов, сортировка вручную
pairs = []
for t in ts:
    for s in (0, 1):
        pr = naive_drift(t, h)[s]
        pairs.append((t, s, abs(WV_L[s][t] - pr)))
pool_nav = sorted(p[2] for p in pairs)
k = math.ceil((len(pool_nav) + 1) * 0.8)
k = min(max(k, 1), len(pool_nav))
q80n = pool_nav[k - 1]
print("наивный пул |res| (t,ряд,val):")
for t, s, v in pairs:
    print(f"  t={t} ряд={'AB'[s]}  |res|={v:.6f}")
print(f"sorted: {[f'{v:.6f}' for v in pool_nav]}")
print(f"n={len(pool_nav)}, k=ceil({len(pool_nav)+1}*0.8)={k}, q80={q80n:.6f}")
assert abs(q80v - q80n) < 1e-12, (q80v, q80n)
covered = []
for s in (0, 1):
    pr = naive_drift(tm, h)[s]
    lo, hi = math.expm1(max(pr - q80n, 0.0)), math.expm1(pr + q80n)
    y = RUB[s][tm]
    cov = lo <= y <= hi
    covered.append(cov)
    print(f"ряд {'AB'[s]}: pred_log={pr:.6f} -> [{lo:.1f}; {hi:.1f}] руб, "
          f"y={y}, |res_log|={abs(math.log1p(y)-pr):.6f} <= q80? {cov}")
# сверка с векторным предсказанием/полосой
for s in (0, 1):
    lo_v = math.expm1(max(pred_v[s] - q80v, 0.0))
    hi_v = math.expm1(pred_v[s] + q80v)
    lo_n = math.expm1(max(naive_drift(tm, h)[s] - q80n, 0.0))
    hi_n = math.expm1(naive_drift(tm, h)[s] + q80n)
    assert abs(lo_v - lo_n) < 1e-9 and abs(hi_v - hi_n) < 1e-9
print(f"покрытие 2 ячеек: {sum(covered)}/2 (вектор==наивный)  OK")

# ---------------------------------------------------------- 5) покрытие счётчик
print("=== 5) счётчик покрытия на сетке origins 13..16, h=1 (ручная сверка) ===")
cnt, tot = 0, 0
for oo in range(13, 17):
    ttm = oo  # h=1
    tss, _, _ = champ_iv.select_pool(oo, 1)
    pl = sorted(abs(WV_L[s][t] - naive_drift(t, 1)[s]) for t in tss for s in (0, 1))
    kk = min(max(math.ceil((len(pl) + 1) * 0.8), 1), len(pl))
    qq = pl[kk - 1]
    for s in (0, 1):
        pr = naive_drift(ttm, 1)[s]
        cov = abs(math.log1p(RUB[s][ttm]) - pr) <= qq + 1e-15
        cnt += cov
        tot += 1
print(f"наивный счёт: {cnt}/{tot} = {cnt/tot:.3f}")
assert tot == 8
print("все проверки пройдены")
