"""Статистическая значимость сравнения моделей (предрегистрировано в configs/validation.yaml,
реализовано 06.10 после аудита rc0: до этого момента процедур не существовало).

Протокол (B3 §3):
- h ∈ {1,3,6}: Diebold–Mariano на кросс-секционно усреднённых loss differentials
  по origin'ам, HAC-дисперсия (Newey–West, лаг h−1) + поправка Harvey–Leybourne–Newbold.
  Плюс парный Wilcoxon по рядам (per-series MAE) как кросс-секционный контроль.
- h = 12 (1 origin): DM невозможен по построению → парный t / Wilcoxon по 1947 МО
  + 95% CI разности медиан region-cluster bootstrap (10 000 ресэмплов по регионам).
- Множественные сравнения: Holm внутри каждого горизонта.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _per_origin_diff(errors_a: pd.Series, errors_b: pd.Series, h: int) -> np.ndarray:
    """Loss differential |e_a|−|e_b| по ячейкам (series, origin), усреднение по кросс-секции."""
    d = (errors_a - errors_b).dropna()
    by_origin = d.groupby(level="origin").mean()  # D̄_t: гасит контемпоральную кросс-корреляцию
    return by_origin.sort_index().values


def dm_test(errors_a: pd.Series, errors_b: pd.Series, h: int) -> dict:
    """DM + Newey-West(lag=h−1) + Harvey–Leybourne–Newbold для малых выборок.
    errors_*: Series с MultiIndex (origin, series_id) абсолютных ошибок.
    Возвращает stat, p (двусторонний), n_origins, mean_diff."""
    D = _per_origin_diff(errors_a, errors_b, h)
    n = len(D)
    if n < 3:
        return {"stat": np.nan, "p": np.nan, "n_origins": n, "mean_diff": float(np.mean(D)) if n else np.nan}
    mean_d = D.mean()
    # Newey-West дисперсия среднего: γ0 + 2·Σ_{k=1}^{h-1} w_k γ_k, веса Бартлетта
    def autocov(x, k):
        return np.mean((x[k:] - x.mean()) * (x[:-k] - x.mean()))
    var = autocov(D, 0)
    for k in range(1, h):
        var += 2 * (1 - k / h) * autocov(D, k)
    var /= n
    if var <= 0:
        return {"stat": np.nan, "p": np.nan, "n_origins": n, "mean_diff": float(mean_d)}
    dm = mean_d / np.sqrt(var)
    # Поправка HLN (Harvey, Leybourne & Newbold 1997): для h-шагового прогноза
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * adj if adj > 0 else dm
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"stat": float(stat), "p": float(p), "n_origins": n, "mean_diff": float(mean_d)}


def paired_wilcoxon(mae_a: pd.Series, mae_b: pd.Series) -> dict:
    """Парный Wilcoxon signed-rank по per-series MAE (Series, index=series_id)."""
    d = (mae_a - mae_b).dropna()
    d = d[d != 0]
    if len(d) < 10:
        return {"p": np.nan, "n": len(d), "median_diff": float(d.median()) if len(d) else np.nan}
    p = stats.wilcoxon(d).pvalue
    return {"p": float(p), "n": int(len(d)), "median_diff": float(d.median())}


def region_cluster_bootstrap_ci(mae_a: pd.Series, mae_b: pd.Series, regions: pd.Series,
                                n_boot: int = 10_000, seed: int = 42) -> dict:
    """95% CI медианы (mae_a − mae_b) с ресэмплингом РЕГИОНОВ (МО внутри региона
    коррелированы — кластерный бутстрэп, B3 §3.1)."""
    df = pd.concat([mae_a.rename("a"), mae_b.rename("b"), regions.rename("region")], axis=1).dropna()
    df["d"] = df["a"] - df["b"]
    by_region = {r: g["d"].values for r, g in df.groupby("region")}
    region_names = np.array(list(by_region))
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(region_names, size=len(region_names), replace=True)
        meds[i] = np.median(np.concatenate([by_region[r] for r in pick]))
    return {"median_diff": float(df["d"].median()),
            "ci_lo": float(np.percentile(meds, 2.5)),
            "ci_hi": float(np.percentile(meds, 97.5))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-поправка по семейству. Вход {имя: p}, выход {имя: p_holm}."""
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    m = len(items)
    ordered = sorted(items, key=lambda kv: kv[1])
    out, running = {}, 0.0
    for rank, (k, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = running
    for k, v in pvals.items():
        if k not in out:
            out[k] = np.nan
    return out


def effect_sizes(mae_a: pd.Series, mae_b: pd.Series) -> dict:
    """Effect size рядом с p: медиана отношения и медиана разности (руб)."""
    j = pd.concat([mae_a.rename("a"), mae_b.rename("b")], axis=1).dropna()
    ratio = (j["a"] / j["b"]).median()
    return {"median_ratio": float(ratio), "median_diff_rub": float((j["a"] - j["b"]).median())}
