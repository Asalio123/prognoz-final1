"""rc1 / O6: sensitivity пермутационного теста burst×CPD к ширине окна совпадения.

База — точная реплика scripts/15_run_news_cpd.py (та же нуль-модель circular-shift,
тот же ряд, тот же PELT l2 pen=5, те же burst-метки z>1.65 из news_monthly_v2,
то же окно оценки 2023-01..2024-12, тот же seed 42).
Меняется ТОЛЬКО полуширина окна совпадения k: ±0 (строгий месяц), ±1 (отчётная), ±2.

Подача: вся сетка целиком, НЕ cherry-pick лучшего p.
Контроль реплики: при k=±1 и 10 000 reps должно воспроизвестись observed=3, p≈0.1335.
Дополнительно для k=±1 прогон ×100 000 (Monte-Carlo ошибка p).

Выход: outputs/rc1_o6_perm_sensitivity.csv, outputs/rc1_o6_perm_sensitivity.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(__file__).resolve().parents[1]
PEN = 5.0
WINDOW = ("2023-01", "2024-12")
GRID_K = [0, 1, 2]
N_REPS = 10_000
N_REPS_BIG = 100_000  # только для k=1 (отчётное окно)


def observed_count(topics, bursts, cps, all_months, pos, k):
    """Число совпадений CP∧burst при полуширине окна k (логика 15_run_news_cpd)."""
    cnt = 0
    for t in topics:
        for m in all_months:
            if m not in cps:
                continue
            i = pos[m]
            near = {all_months[j] for j in range(max(0, i - k), min(len(all_months), i + k + 1))}
            if bursts[t] & near:
                cnt += 1
    return cnt


def null_distribution(topics, bursts, cps, all_months, pos, k, n_reps, seed, symmetric):
    """Circular-shift нуль-модель: сдвиг burst-меток каждой темы на случайный сдвиг,
    каждый rep от ИСТИННЫХ burst'ов (как в 15_run_news_cpd.py).

    symmetric=False: дословная реплика 15_run_news_cpd.py — в нуле сдвинутый burst
    засчитывается только ТОЧНЫМ попаданием в месяц CP, тогда как observed засчитывает
    окно ±k. Асимметрия => нуль занижен (~в 2k+1 раза), p антиконсервативно.
    symmetric=True: та же нуль-модель, но совпадение в нуле засчитывается тем же
    окном ±k, что и в observed — методологически корректный вариант.
    """
    rng = np.random.default_rng(seed)
    n_m = len(all_months)
    null = np.empty(n_reps, dtype=int)
    for rep in range(n_reps):
        cnt = 0
        for t in topics:
            shift = int(rng.integers(1, n_m))
            shifted = {all_months[(pos[m] + shift) % n_m] for m in bursts[t] if m in pos}
            for m in all_months:
                if m not in cps:
                    continue
                if symmetric:
                    i = pos[m]
                    near = {all_months[j] for j in range(max(0, i - k), min(n_m, i + k + 1))}
                    if shifted & near:
                        cnt += 1
                elif m in shifted:
                    cnt += 1
        null[rep] = cnt
    return null


def main():
    nm = pd.read_parquet(ROOT / "outputs" / "news_monthly_v2.parquet")
    topics = [c[2:] for c in nm.columns if c.startswith("z_")]
    bursts = {t: set(nm.index[nm[f"z_{t}"] > 1.65]) for t in topics}

    g = pd.read_parquet(ROOT / "data" / "external" / "sberindex_api" / "consumer-spending-growth_parquet.parquet")
    g["month"] = pd.to_datetime(g["period"]).dt.strftime("%Y-%m")
    nat = g[(g["type"] == "Всего") & (g["value_type"] == "Номинальное")].set_index("month")["value"].sort_index()
    nat = nat[~nat.index.duplicated()]
    assert len(nat) == 93

    y = nat.to_numpy(dtype=float)
    bk = [b for b in rpt.Pelt(model="l2").fit(y).predict(pen=PEN) if b < len(y)]
    cp_all = {nat.index[b] for b in bk}
    cps = {m for m in cp_all if WINDOW[0] <= m <= WINDOW[1]}
    all_months = sorted(m for m in nat.index if WINDOW[0] <= m <= WINDOW[1])
    assert len(all_months) == 24
    pos = {m: i for i, m in enumerate(all_months)}
    print(f"CP в окне: {sorted(cps)}")
    for t in topics:
        print(f"bursts {t}: {sorted(bursts[t])}")

    rows = []
    for k in GRID_K:
        obs = observed_count(topics, bursts, cps, all_months, pos, k)
        for variant, sym in (("as_reported", False), ("symmetric", True)):
            null = null_distribution(topics, bursts, cps, all_months, pos, k, N_REPS, seed=42, symmetric=sym)
            p = float(np.mean(null >= obs))
            rows.append(dict(window=f"±{k}", k=k, variant=variant, reps=N_REPS, observed=obs,
                             null_mean=round(float(null.mean()), 3),
                             null_sd=round(float(null.std()), 3), p=round(p, 4)))
            print(f"k=±{k} [{variant}]: obs={obs}, null {null.mean():.2f}±{null.std():.2f}, p={p:.4f}")

    # расширенный прогон для отчётного окна k=±1, оба варианта
    k = 1
    obs = observed_count(topics, bursts, cps, all_months, pos, k)
    mc = {}
    for variant, sym in (("as_reported", False), ("symmetric", True)):
        null_big = null_distribution(topics, bursts, cps, all_months, pos, k, N_REPS_BIG, seed=42, symmetric=sym)
        p_big = float(np.mean(null_big >= obs))
        mc[variant] = (p_big, float(np.sqrt(p_big * (1 - p_big) / N_REPS_BIG)))
        rows.append(dict(window=f"±{k}", k=k, variant=variant, reps=N_REPS_BIG, observed=obs,
                         null_mean=round(float(null_big.mean()), 3),
                         null_sd=round(float(null_big.std()), 3), p=round(p_big, 4)))
        print(f"k=±1 [{variant}] ×{N_REPS_BIG}: p={p_big:.4f} (MC-SE {mc[variant][1]:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / "rc1_o6_perm_sensitivity.csv", index=False)

    md = ["# Sensitivity: burst×CPD пермутационный тест, сетка окон совпадения (rc1/O6)", "",
          "База: точная реплика 15_run_news_cpd.py (ряд consumer-spending-growth «Всего», Номинальное,",
          "93 мес; PELT l2 pen=5, CP в окне: " + ", ".join(sorted(cps)) + "; burst = z>1.65 из",
          "news_monthly_v2, 4 активные темы; нуль-модель circular-shift, seed 42).",
          "Меняется только полуширина окна совпадения k. Сетка подаётся целиком, без выбора лучшего p.",
          "",
          "Два варианта нуль-модели:",
          "- **as_reported** — дословная реплика 15_run_news_cpd.py: в пермутациях сдвинутый burst",
          "  засчитывается ТОЛЬКО точным попаданием в месяц CP, тогда как observed — окном ±k.",
          "  Асимметрия занижает нуль (~в 2k+1 раза) => p антиконсервативно (найдено в rc1/O6).",
          "- **symmetric** — та же circular-shift нуль-модель, но совпадение в пермутациях",
          "  засчитывается тем же окном ±k, что и observed. Методологически корректный вариант.", "",
          "| окно | нуль-модель | reps | observed CP∧burst | null mean±sd | p |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['window']} мес | {r['variant']} | {r['reps']:,} | {r['observed']} | "
                  f"{r['null_mean']}±{r['null_sd']} | {r['p']:.4f} |")
    md += ["", f"Monte-Carlo SE при ×{N_REPS_BIG:,} (k=±1): as_reported ±{mc['as_reported'][1]:.4f}, "
           f"symmetric ±{mc['symmetric'][1]:.4f}.",
           "", "## Интерпретация", ""]
    (ROOT / "outputs" / "rc1_o6_perm_sensitivity.md").write_text("\n".join(md))
    print("\n" + "\n".join(md[16:26]))


if __name__ == "__main__":
    main()
