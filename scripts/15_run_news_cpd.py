"""Новости × CPD: матрица согласования (К5, союз «и» — обязательная вторая половина).

Burst detection на новостных потоках (expanding z, min-count гейт из таксономии v3)
× PELT (model='l2', pen=5) на национальном ряде потребления.
Матрица: CP∧burst = национальный шок / CP без burst = локальный / burst без CP = проверка.
Значимость совпадений: circular-shift пермутации ×10 000 (месяцы коррелированы).

Исправления v2 (06.10.2026, аудит rc0 P0-1 — двойной баг):
БЫЛО: PELT на consumper-spending-index-sa БЕЗ фильтра type -> склейка 5 категорий
(465 строк, 5 уровней помесячно) давала ложные CP {2020-04, 2020-07, 2024-01};
окно ±1 ломалось в одностороннее из-за .index() на продублированном списке месяцев
(120 элементов вместо 24, матрица 600 строк); пермутации накапливались in-place.
СТАЛО: ряд = consumer-spending-growth, type=='Всего', value_type=='Номинальное'
(панель номинальная — C3), 93 мес 2018-12..2026-08. Обоснование выбора ряда:
на уровне SA (log, type=='Всего') PELT при pen∈{1,2,5,10} находит 0 CP за весь
2018–2026, включая COVID-провал 103.7→73.1 — ряд слишком гладкий, тест бессилен
(проверено перезапуском 06.10.2026). Структурные сдвиги живут в темпах г/г, а не
в десезонированном уровне; новостные всплески тоже отвечают на изменения, не на
уровни. pen=5: на номинальном ряде множество CP идентично при pen∈{5,10}
(плато нечувствительности); pen=2 добавляет 2021-11, 2025-08, 2026-06 (вне окна).
Реальный ряд (робастность): CP пересекаются с номинальным {2023-02, 2023-07,
2024-05, 2024-10} + 2023-12.
Выход: outputs/news_cpd_matrix_v2.csv, outputs/news_cpd_perm_v2.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(__file__).resolve().parents[1]
PEN = 5.0  # плато нечувствительности 5–10 на номинальном ряде (см. docstring)
N_REPS = 10_000  # R07 §4: circular-shift ×10 000 (было 500 — расход со спекой)
WINDOW = ("2023-01", "2024-12")


def main():
    nm = pd.read_parquet(ROOT / "outputs" / "news_monthly_v2.parquet")
    topics = [c[2:] for c in nm.columns if c.startswith("z_")]
    bursts = {}
    for t in topics:
        z = nm[f"z_{t}"]
        bursts[t] = set(nm.index[z > 1.65])  # месяцы с тематическим всплеском (z уже NaN-гейтнут по count)
        print(f"bursts {t}: {sorted(bursts[t])}")

    # национальный месячный ряд: темпы г/г «Всего», номинальные (панель номинальная — C3)
    g = pd.read_parquet(ROOT / "data" / "external" / "sberindex_api" / "consumer-spending-growth_parquet.parquet")
    g["month"] = pd.to_datetime(g["period"]).dt.strftime("%Y-%m")
    nat = g[(g["type"] == "Всего") & (g["value_type"] == "Номинальное")].set_index("month")["value"].sort_index()
    nat = nat[~nat.index.duplicated()]  # страховка от склеек
    assert len(nat) == 93, f"ожидался чистый национальный ряд 93 мес, получено {len(nat)}"

    y = nat.to_numpy(dtype=float)
    bk = [b for b in rpt.Pelt(model="l2").fit(y).predict(pen=PEN) if b < len(y)]
    cp_all = {nat.index[b] for b in bk}
    cps = {m for m in cp_all if WINDOW[0] <= m <= WINDOW[1]}
    print(f"\nPELT(l2, pen={PEN}) на growth «Всего» (номин.): все CP = {sorted(cp_all)}")
    print(f"CP в окне {WINDOW[0]}..{WINDOW[1]}: {sorted(cps)}")

    # матрица совпадений по месяцам (окно ±1, месяцы дедуплицированы — починка бага B)
    all_months = sorted(m for m in nat.index if WINDOW[0] <= m <= WINDOW[1])
    assert len(all_months) == 24 and len(set(all_months)) == 24
    pos = {m: i for i, m in enumerate(all_months)}  # позиция месяца, без .index() по дублям
    rows = []
    for t, bursts_m in bursts.items():
        for m in all_months:
            i = pos[m]
            near = {all_months[j] for j in range(max(0, i - 1), min(len(all_months), i + 2))}
            rows.append(dict(category="Все категории (нац., г/г)", topic=t, month=m,
                             cp=m in cps, burst=bool(bursts_m & near)))
    mx = pd.DataFrame(rows)
    mx["cell"] = np.select([mx.cp & mx.burst, mx.cp & ~mx.burst, ~mx.cp & mx.burst],
                           ["CP∧news (нац. шок)", "CP без новостей (локальный)", "news без CP"],
                           default="—")
    mx.to_csv(ROOT / "outputs" / "news_cpd_matrix_v2.csv", index=False)
    assert len(mx) == 24 * len(topics), f"матрица {len(mx)} строк вместо {24 * len(topics)}"

    piv = mx[mx.cp | mx.burst].groupby(["topic", "cell"]).size().unstack(fill_value=0)
    print(f"\nматрица {len(mx)} строк ({len(topics)} темы × 24 мес):")
    print(piv.to_string())

    # circular-shift пермутация: сдвигаем burst-метки, каждый rep от ИСТИННЫХ burst'ов
    rng = np.random.default_rng(42)
    obs = int((mx.cp & mx.burst).sum())
    n_m = len(all_months)
    null = []
    for rep in range(N_REPS):
        shifted_bursts = {}
        for t in topics:
            shift = int(rng.integers(1, n_m))
            shifted_bursts[t] = {all_months[(pos[m] + shift) % n_m] for m in bursts[t] if m in pos}
        cnt = 0
        for t in topics:
            sub = mx[mx.topic == t]
            for m, cp in zip(sub["month"], sub["cp"]):
                if cp and m in shifted_bursts[t]:
                    cnt += 1
        null.append(cnt)
    null = np.array(null)
    p = float(np.mean(null >= obs))
    with open(ROOT / "outputs" / "news_cpd_perm_v2.txt", "w") as f:
        f.write(f"series: consumer-spending-growth, type=Всего, Номинальное, 2018-12..2026-08 (93 мес)\n")
        f.write(f"PELT l2 pen={PEN} (плато 5–10); CP все: {sorted(cp_all)}\n")
        f.write(f"CP в окне: {sorted(cps)}; темы: {topics}\n")
        f.write(f"observed CP∧burst: {obs}\nnull mean: {null.mean():.2f}, p(circular-shift, {N_REPS} reps) = {p:.4f}\n")
    print(f"\nCP∧burst наблюдаемо: {obs}, null {null.mean():.2f}, p={p:.4f} ({N_REPS} reps)")


if __name__ == "__main__":
    main()
