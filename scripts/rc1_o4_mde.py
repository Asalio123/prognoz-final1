"""Мощность нуля (MDE) для трёх отрицательных результатов К3.

Превращает «не нашли» в «если бы эффект был >= X, мы бы его увидели с мощностью 80%».
Только готовые артефакты outputs/, ничего не перегоняет.

(a) Пространственное распространение (outputs/spatial_propagation.csv):
    нуль = circular-shift пермутации (1000 reps), счётчик связанных пар при n=42404 пар.
    Аппроксимация нуля Poisson(mu0) (редкие события; заявлено как допущение).
    Критическое значение c = min{k: P(Pois(mu0) >= k) <= 0.05} (односторонний тест,
    как в скрипте 14). MDE-лифт lambda* = min{lambda: P(Pois(lambda*mu0) >= c) >= 0.80}.
    Кросс-чек: нормальная аппроксимация разности долей при p0 = mu0/42404.

(b) Refit-от-брейка (outputs/downstream_dmae_rows_v2.parquet, legacy_pen3, h=3,
    fallback=False — строка «+1.1%» отчёта): парные d_i = ae_refit/ae_full - 1.
    MDE = (z_0.975 + z_0.80) * sd(d)/sqrt(n) — двусторонний alpha=0.05, мощность 80%.
    Считается на ячейках (n=98) и на посерийных средних (n=54, контроль корреляции
    ячеек одного ряда).

(c) burst x CPD (outputs/news_cpd_perm_v2.txt: observed 3, ожидание 1.35,
    p=0.1335 по 10000 circular-shift): нуль ~ Poisson(1.35).
    (i) какой observed count дал бы p<0.05; (ii) MDE истинного rate при мощности 80%.
    Валидация аппроксимации: P(Pois(1.35) >= 3) должна быть близка к 0.1335.

Выход: outputs/rc1_o4_mde.csv + консоль.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
POWER = 0.80
Z_A1 = stats.norm.ppf(1 - ALPHA)        # 1.645 односторонний
Z_A2 = stats.norm.ppf(1 - ALPHA / 2)    # 1.960 двусторонний
Z_B = stats.norm.ppf(POWER)             # 0.842


def poisson_critical(mu0, alpha=ALPHA):
    """min k: P(Pois(mu0) >= k) <= alpha (односторонний)."""
    k = int(np.ceil(mu0))
    while stats.poisson.sf(k - 1, mu0) > alpha:
        k += 1
    return k


def poisson_mde_lift(mu0, c, power=POWER):
    """min lambda: P(Pois(lambda*mu0) >= c) >= power (сетка 0.001)."""
    lam = np.arange(1.0, 30.0, 0.001)
    pw = stats.poisson.sf(c - 1, lam * mu0)
    idx = np.argmax(pw >= power)
    return float(lam[idx]) if pw[idx] >= power else np.nan


def main():
    rows = []

    # ---------------- (a) пространственное распространение ---------------- #
    sp = pd.read_csv(ROOT / "outputs" / "spatial_propagation.csv")
    print("=== (a) spatial propagation, n_pairs=42404, односторонний alpha=0.05 ===")
    for _, r in sp.iterrows():
        mu0, obs, n_pairs = r.null_mean, r.observed_linked, r.pairs_le80km
        if mu0 <= 0:
            rows.append(dict(block="spatial", method=r.method, n_pairs=n_pairs,
                             null_mean=mu0, observed=obs, note="нуль вырожден (0 ожидаемых) — MDE не определён"))
            continue
        c = poisson_critical(mu0)
        lam = poisson_mde_lift(mu0, c)
        p0 = mu0 / n_pairs
        # нормальная аппроксимация: одна доля против фиксированной p0 (нуль-пермутации)
        mde_rate = (Z_A1 + Z_B) * np.sqrt(p0 * (1 - p0) / n_pairs)
        lift_norm = (p0 + mde_rate) / p0
        rows.append(dict(block="spatial", method=r.method, n_pairs=n_pairs, null_mean=mu0,
                         observed=obs, p0_per_pair=p0, crit_count=c,
                         mde_lift_poisson=round(lam, 2),
                         mde_count=round(lam * mu0, 1),
                         mde_lift_norm=round(lift_norm, 2)))
        print(f"{r.method}: mu0={mu0}, obs={obs}, crit={c}, MDE-lift Poisson={lam:.2f} "
              f"(=> {lam*mu0:.1f} связанных пар), norm-кроссчек={lift_norm:.2f}")

    # ---------------- (b) refit-от-брейка ---------------- #
    d = pd.read_parquet(ROOT / "outputs" / "downstream_dmae_rows_v2.parquet")
    sub = d[(d.setting == "legacy_pen3") & (d.horizon == 3) & (~d.fallback)].copy()
    n_cells = len(sub)
    # каноническая per-series дельта скрипта 13: суммарное отношение по ряду
    # (устойчива к малым знаменателям отдельных ячеек, убирает внутрисерийную
    # корреляцию origins агрегацией)
    per_series = sub.groupby("series_id").apply(
        lambda x: x.ae_refit.sum() / max(x.ae_full.sum(), 1e-9) - 1, include_groups=False)
    n_ser = len(per_series)
    pooled = sub.ae_refit.mean() / sub.ae_full.mean() - 1  # якорь: должно дать +1.10%
    sd_s = per_series.std(ddof=1)
    mean_s = per_series.mean()
    mde_ser = (Z_A2 + Z_B) * sd_s / np.sqrt(n_ser)
    t_ser = mean_s / (sd_s / np.sqrt(n_ser))
    p_ser = 2 * stats.t.sf(abs(t_ser), n_ser - 1)
    rows.append(dict(block="downstream_byseries", setting="legacy_pen3 h=3 noFB",
                     n_series=n_ser, n_cells=n_cells,
                     pooled_pct=round(pooled * 100, 2),
                     mean_series_pct=round(mean_s * 100, 2),
                     median_series_pct=round(per_series.median() * 100, 2),
                     sd_series_pct=round(sd_s * 100, 2),
                     t_obs=round(t_ser, 3), p_obs=round(p_ser, 3),
                     mde_abs_pct=round(mde_ser * 100, 2)))
    # sensitivity: ячеечные d_i, винзоризация 1/99% (сырые взорваны малыми ae_full)
    sub["d_rel"] = sub.ae_refit / sub.ae_full - 1.0
    lo, hi = sub.d_rel.quantile([0.01, 0.99])
    dw = sub.d_rel.clip(lo, hi)
    mde_w = (Z_A2 + Z_B) * dw.std(ddof=1) / np.sqrt(n_cells)
    rows.append(dict(block="downstream_cells_w199", setting="legacy_pen3 h=3 noFB",
                     n_cells=n_cells, mean_pct=round(dw.mean() * 100, 2),
                     sd_pct=round(dw.std(ddof=1) * 100, 2),
                     mde_abs_pct=round(mde_w * 100, 2),
                     note="винзоризация 1/99%; сырые ячеечные отношения вырождены малыми знаменателями"))
    # 10%-триммированная посерийная дельта (масштаб «типичного» ряда)
    tr = stats.trim_mean(per_series, 0.1)
    lo10, hi10 = per_series.quantile([0.1, 0.9])
    sd_tr = per_series.clip(lo10, hi10).std(ddof=1)
    mde_tr = (Z_A2 + Z_B) * sd_tr / np.sqrt(n_ser)
    rows.append(dict(block="downstream_byseries_trim10", setting="legacy_pen3 h=3 noFB",
                     n_series=n_ser, trimmed_mean_pct=round(tr * 100, 2),
                     sd_pct=round(sd_tr * 100, 2), mde_abs_pct=round(mde_tr * 100, 2)))
    # знаковый тест: доля ячеек, где refit лучше (робастно к хвостам, «разность долей»)
    share = float((sub.ae_refit < sub.ae_full).mean())
    mde_share = (Z_A2 + Z_B) * np.sqrt(0.25 / n_cells)
    share_s = float((per_series < 0).mean())
    mde_share_s = (Z_A2 + Z_B) * np.sqrt(0.25 / n_ser)
    rows.append(dict(block="downstream_sign_cells", setting="legacy_pen3 h=3 noFB",
                     n=n_cells, share_refit_better=round(share, 4),
                     mde_share_pp=round(mde_share * 100, 1),
                     note="MDE отклонения доли от 0.50, двуст. alpha=0.05, мощность 80%"))
    rows.append(dict(block="downstream_sign_series", setting="legacy_pen3 h=3 noFB",
                     n=n_ser, share_refit_better=round(share_s, 4),
                     mde_share_pp=round(mde_share_s * 100, 1)))
    print(f"\n=== (b) downstream refit ===\npooled ΔMAE={pooled*100:+.2f}% (якорь отчёта +1.10%); "
          f"рядов={n_ser}, ячеек={n_cells}")
    print(f"посерийно: mean={mean_s*100:+.2f}%, sd={sd_s*100:.2f}%, t={t_ser:.2f}, "
          f"p={p_ser:.3f}, MDE(80%, двуст.)=±{mde_ser*100:.2f} п.п.")
    print(f"trim10 посерийно: mean={tr*100:+.2f}%, MDE=±{mde_tr*100:.2f} п.п.; "
          f"ячейки w1/99: MDE=±{mde_w*100:.2f} п.п.")
    print(f"знаковый: доля refit лучше = {share:.3f} (n={n_cells}), MDE=±{mde_share*100:.1f} п.п. "
          f"от 50%; посерийно {share_s:.3f} (n={n_ser}), MDE=±{mde_share_s*100:.1f} п.п.")

    # ---------------- (c) burst x CPD ---------------- #
    mu0_c, obs_c = 1.35, 3
    p_check = stats.poisson.sf(obs_c - 1, mu0_c)
    crits = {k: stats.poisson.sf(k - 1, mu0_c) for k in range(3, 8)}
    c_c = poisson_critical(mu0_c)
    lam_c = poisson_mde_lift(mu0_c, c_c)
    rows.append(dict(block="burst_cpd", null_mean=mu0_c, observed=obs_c,
                     p_poisson_obs=round(p_check, 4), p_perm=0.1335,
                     crit_count=c_c, p_at_crit=round(crits[c_c], 4),
                     mde_true_rate=round(lam_c * mu0_c, 2), mde_lift=round(lam_c, 2)))
    print(f"\n=== (c) burst x CPD ===\nP(Pois(1.35)>=3)={p_check:.4f} (пермутации: 0.1335)")
    print("P(>=k):", {k: round(v, 4) for k, v in crits.items()})
    print(f"min count для p<0.05: {c_c}; MDE истинного rate: {lam_c*mu0_c:.2f} "
          f"(лифт x{lam_c:.2f} над ожиданием 1.35)")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "rc1_o4_mde.csv", index=False)
    print("\nзаписано outputs/rc1_o4_mde.csv")


if __name__ == "__main__":
    main()
