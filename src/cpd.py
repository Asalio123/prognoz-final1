"""Общий CPD-слой (К3, R04): десезонирование, детекторы, инъекции, метрики.

Фиксы по аудиту A11/A05 (review_rc0):
- десезонирование: месячные факторы = шринкаж per-series медианы к покатегорийному
  профилю (w = n/(n+k), n=2 наблюдения на месяц, k=2) вместо median-of-2, которая
  поглощала level-shift и зеркалила его в соседний год (фантомы, аттенюация сигнала);
- BOCPD: корректная рекурсия Adams–MacKay (рост r→r+1, сброс→0 с prior predictive,
  Student-t predictive из normal-gamma обновления) вместо сдвинутой матрицы;
- CUSUM: масштаб по MAD приростов вместо std (тяжёлые хвосты раздували std → FAR
  не опускался ниже 53/1000);
- все пороги — абсолютные, калибруются на чистой панели и замораживаются (никаких
  самоссылочных квантилей собственного ряда).
"""
from __future__ import annotations

import numpy as np
import ruptures as rpt
from scipy.special import gammaln

N_MONTHS = 24

# Замороженная конфигурация предобработки (выбрана свипом на калибровочной панели,
# см. review_rc0/FIX_A11_report.md): ridge-шринкаж k=4 к профилю категории (k=4 даёт
# аттенюацию инъекции 1.24σ против 1.11σ при k=2 — меньше поглощение брейка),
# совместная оценка с трендом, Huber-IRLS против поглощения level-shift.
K_SHRINK = 4.0
DETREND = True
ROBUST = True


# --------------------------------------------------------------------------- #
# Десезонирование
# --------------------------------------------------------------------------- #

def deseasonalize_old(y: np.ndarray) -> np.ndarray:
    """Старая схема (median-of-2) — оставлена для сравнений до/после."""
    y = np.asarray(y, dtype=float)
    prof = np.array([np.median(y[m::12]) for m in range(12)])
    return y - prof[np.arange(len(y)) % 12]


def category_profiles(logmat: np.ndarray, cats: np.ndarray) -> dict:
    """Покатегорийный сезонный профиль: медиана отклонений от среднего ряда.

    logmat: (n_series, 24) log-уровни; cats: (n_series,) метки категорий.
    Профиль центрирован (среднее по 12 месяцам = 0) — уровень не трогаем.
    """
    logmat = np.asarray(logmat, float)
    dev = logmat - logmat.mean(axis=1, keepdims=True)
    cal = np.arange(logmat.shape[1]) % 12
    profs = {}
    for c in np.unique(cats):
        d = dev[cats == c]  # (n_c, 24)
        p = np.array([np.median(d[:, cal == m]) for m in range(12)])
        profs[c] = p - p.mean()
    return profs


def _robust_slope(y: np.ndarray) -> float:
    """Запасной оценщик дрейфа (не используется в основном пути после перехода на
    совместную ridge-оценку; оставлен для сравнений)."""
    t = np.arange(len(y), dtype=float)
    t = t - t.mean()
    return float(np.dot(t, y) / np.dot(t, t))


def _ridge_deseasonalize(y: np.ndarray, cat_prof: np.ndarray, k: float = 2.0,
                         huber_c: float | None = 1.345, n_irls: int = 4) -> np.ndarray:
    """Совместная оценка: y_t = a + b·t + f[m(t)] + ε с ridge-шринкажем f к
    покатегорийному профилю (штраф k на месячный дамми = k «наблюдений» при n=2 →
    w = n/(n+k) = 0.5, тот же смысл, что у формулы w·медиана+(1−w)·профиль).

    Почему не последовательное «факторы → детренд → факторы»: вторая оценка
    факторов поглощает остатки level-shift (медианы пост-брейковых месяцев тянутся
    за скачком) — измерено: инъекция 1.6σ ослабевала до 0.8–1.0σ.

    huber_c: IRLS с Huber-весами по остаткам (σ̂ = MAD) — месячные дамми и тренд не
    догоняют скачок: пост-брейковые точки получают вес <1, аттенюация инъекции
    снижается (0.86σ → ~1.2σ при k=2). None = чистый ridge.
    """
    y = np.asarray(y, float)
    n = len(y)
    t = (np.arange(n) - (n - 1) / 2) / n  # центрированный, нормированный — стабильнее
    cal = np.arange(n) % 12
    X = np.zeros((n, 14))
    X[:, 0] = 1.0
    X[:, 1] = t
    X[np.arange(n), 2 + cal] = 1.0
    pen = np.zeros(14)
    pen[2:] = k
    tgt = np.zeros(14)
    tgt[2:] = k * cat_prof
    A = X.T @ X + np.diag(pen)
    b = X.T @ y + tgt
    beta = np.linalg.solve(A, b)
    if huber_c is not None:
        w = np.ones(n)
        for _ in range(n_irls):
            r = y - X @ beta
            s = 1.4826 * np.median(np.abs(r - np.median(r)))
            if s < 1e-9:
                break
            u = r / (huber_c * s)
            w = 1.0 / np.maximum(1.0, np.abs(u))
            Xw = X * w[:, None]
            A = Xw.T @ X + np.diag(pen)
            b = Xw.T @ y + tgt
            beta = np.linalg.solve(A, b)
    return y - X @ beta


def deseasonalize(y: np.ndarray, cat_prof: np.ndarray, k: float = 2.0,
                  detrend: bool = True, robust: bool = True) -> np.ndarray:
    """Десезонированный ряд: совместная ridge/Huber-оценка (тренд + месячные факторы
    со шринкажем к профилю категории). detrend=False — только факторы (legacy-путь)."""
    if detrend:
        return _ridge_deseasonalize(y, cat_prof, k=k, huber_c=1.345 if robust else None)
    y = np.asarray(y, float)
    return y - seasonal_factor(y, cat_prof, k=k)[np.arange(len(y)) % 12]


def seasonal_factor(y: np.ndarray, cat_prof: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Месячный фактор ряда: шринкаж per-series медианы к покатегорийному профилю.

    w = n/(n+k): при n=2 наблюдениях на месяц и k=2 → w=0.5. Серийный сдвиг уровня
    загрязняет только половину фактора (а не весь, как в median-of-2); типичная
    сезонность категории при этом вычитается полностью через профиль.
    """
    y = np.asarray(y, float)
    cal = np.arange(len(y)) % 12
    dev = y - y.mean()
    n_per = np.array([np.sum(cal == m) for m in range(12)], float)
    own = np.array([np.median(dev[cal == m]) for m in range(12)])
    w = n_per / (n_per + k)
    f = w * own + (1 - w) * cat_prof
    return f - f.mean()


# --------------------------------------------------------------------------- #
# Масштаб шума
# --------------------------------------------------------------------------- #

def mad_scale(y: np.ndarray) -> float:
    """Робастная σ по м/м приростам: 1.4826·MAD (fallback на std при вырождении).
    Масштаб ШУМА ДИФФОВ — для калибровки инъекций (δ в этих единицах)."""
    d = np.diff(np.asarray(y, float))
    mad = np.median(np.abs(d - np.median(d)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-9:
        s = np.std(d)
    return float(s if s > 1e-9 else 1.0)


def level_scale(y: np.ndarray) -> float:
    """Робастная σ УРОВНЯ остатков: 1.4826·MAD вокруг медианы. Шкала для CUSUM/EWMA/
    BOCPD: они копят отклонения УРОВНЯ от таргета; diff-MAD при положительной
    автокорреляции остатков в 2–4 раза меньше уровневой σ → FAR взрывался до
    30–130/1000 (замерено на калибровочной панели)."""
    y = np.asarray(y, float)
    mad = np.median(np.abs(y - np.median(y)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-9:
        s = np.std(y)
    return float(s if s > 1e-9 else 1.0)


# --------------------------------------------------------------------------- #
# Детекторы (единый вход: десезонированный лог-ряд; брейк b = сегмент с индекса b)
# --------------------------------------------------------------------------- #

def det_pelt(y, pen):
    return [b for b in rpt.Pelt(model="l2", min_size=3).fit(y).predict(pen=pen) if b < len(y)]


def det_binseg(y, pen):
    return [b for b in rpt.Binseg(model="l2", min_size=3).fit(y).predict(pen=pen) if b < len(y)]


def det_window(y, pen):
    # jump=1 обязателен: дефолт ruptures jump=5 квантует брейки в {5,10,15,20} —
    # именно это давало «3955/6454 брейков в idx15» из аудита, а не только сезонность.
    return [b for b in rpt.Window(width=4, model="rbf", min_size=3, jump=1)
            .fit(y).predict(pen=pen) if b < len(y)]


def det_cusum(y, h):
    """Двусторонний CUSUM, шкала по MAD УРОВНЯ (level_scale); рестарт после тревоги,
    таргет = медиана последних ≤6 точек."""
    s = level_scale(y)
    k, thr = 0.5 * s, h * s
    breaks, sp, sn = [], 0.0, 0.0
    mu = np.median(y[:6])
    for t in range(1, len(y)):
        e = y[t] - mu
        sp, sn = max(0.0, sp + e - k), min(0.0, sn + e + k)
        if sp > thr or abs(sn) > thr:
            breaks.append(t)
            sp, sn = 0.0, 0.0
            mu = np.median(y[max(0, t - 5): t + 1])
    return breaks


def _student_t_logpdf(x, df, loc, scale):
    return (gammaln((df + 1) / 2) - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi) - np.log(scale)
            - (df + 1) / 2 * np.log1p(((x - loc) / scale) ** 2 / df))


def bocpd_prestart(y, hazard=1 / 12):
    """Корректный BOCPD (Adams–MacKay 2007) с Student-t predictive (normal-gamma).

    Рекурсия: рост r→r+1 с predictive рана; сброс → r=0 с PRIOR predictive нового
    рана (p(y_t|∅)), как в оригинальном message passing. Возвращает p_restart[t] =
    P(r_t = 0) для t = 1..n−1 (индекс t = граница перед наблюдением t).
    """
    y = np.asarray(y, float)
    n = len(y)
    s2 = level_scale(y) ** 2
    mu0 = float(np.median(y[:6]))
    kap0, alp0, bet0 = 1.0, 2.0, max(s2, 1e-6)
    logH, log1mH = np.log(hazard), np.log(1 - hazard)

    # параметры по гипотезам run-length (индекс = длина рана)
    mu = np.array([mu0]); kap = np.array([kap0])
    alp = np.array([alp0]); bet = np.array([bet0])
    logR = np.array([0.0])  # log P(r_t, y_1..t), нормируется каждый шаг
    p_restart = np.zeros(n + 1)

    for t in range(1, n + 1):
        x = y[t - 1]
        # predictive существующих ранов для x
        df = 2 * alp
        loc = mu
        scale = np.sqrt(bet * (kap + 1) / (alp * kap))
        lp_run = _student_t_logpdf(x, df, loc, scale)
        # prior predictive (новый ран: данных нет)
        lp_new = _student_t_logpdf(x, 2 * alp0, mu0, np.sqrt(bet0 * (kap0 + 1) / (alp0 * kap0)))
        # рост: r → r+1
        growth = logR + lp_run + log1mH          # growth[r] → новый r+1
        # сброс: H · p_prior(x) · Σ_r P(r, y_1..t-1); logR нормирован → Σ=1 в пространстве вероятностей
        from scipy.special import logsumexp
        cp = logH + lp_new + logsumexp(logR)
        new = np.empty(len(logR) + 1)
        new[0] = cp
        new[1:] = growth
        new -= logsumexp(new)
        logR = new
        # индекс согласован с ruptures: p_restart[b] = «новый сегмент начался с y[b]»
        p_restart[t - 1] = np.exp(logR[0]) if t > 1 else 0.0
        # обновление параметров: ран r поглотил x; новый ран 0 = приор
        kap_n = kap + 1
        mu_n = (kap * mu + x) / kap_n
        alp_n = alp + 0.5
        bet_n = bet + 0.5 * kap * (x - mu) ** 2 / kap_n
        mu = np.concatenate([[mu0], mu_n])
        kap = np.concatenate([[kap0], kap_n])
        alp = np.concatenate([[alp0], alp_n])
        bet = np.concatenate([[bet0], bet_n])
    return p_restart


def det_bocpd(y, thr, hazard=1 / 12):
    """Брейк = P(r_t = 0) > thr, в терминах ruptures-границ (b = новый сегмент с y[b]).
    Порог абсолютный, калибруется на чистой панели (никаких квантилей своего ряда)."""
    p = bocpd_prestart(y, hazard=hazard)
    return [b for b in range(1, len(y)) if p[b] > thr]


def det_ewma(y, L, lam=0.25):
    """EWMA-карта (Roberts 1959), R04 M8: z_t = λy_t + (1−λ)z_{t−1}, тревога при
    |z_t − μ0| > L·σ·√(λ/(2−λ)); μ0 = медиана первых 6, σ = MAD уровня (level_scale).
    Индекс тревоги = месяц пересечения (детекция, не локализация)."""
    s = level_scale(y)
    mu = np.median(y[:6])
    thr = L * s * np.sqrt(lam / (2 - lam))
    z = mu
    breaks = []
    for t in range(1, len(y)):
        z = lam * y[t] + (1 - lam) * z
        if abs(z - mu) > thr:
            breaks.append(t)
            z = y[t]  # рестарт от текущего уровня
            mu = y[t]
    return breaks


# --------------------------------------------------------------------------- #
# Инъекции и метрики
# --------------------------------------------------------------------------- #

def inject(y, kind, tau, delta, s=None, rng=None):
    """Инъекция в лог-ряд ДО десезонирования. delta в единицах σ приростов (s)."""
    y = np.asarray(y, float)
    s = mad_scale(y) if s is None else s
    z = y.copy()
    if kind == "level":
        z[tau:] += delta * s
    elif kind == "trend":
        z[tau:] += delta * s * np.arange(1, len(y) - tau + 1)
    elif kind == "variance":
        rng = np.random.default_rng(0) if rng is None else rng
        z[tau:] += rng.normal(0, (delta - 1.0) * s, len(y) - tau)  # delta = кратность σ
    return z


def eval_detection(found, true_tau, tol=2):
    """tp/fp/fn и delay при окне толерантности ±tol (дубли в окне = один TP)."""
    hits = sorted(f for f in found if abs(f - true_tau) <= tol)
    tp = 1 if hits else 0
    fp = len([f for f in found if abs(f - true_tau) > tol])
    fn = 0 if hits else 1
    delay = (hits[0] - true_tau) if hits else np.nan
    return tp, fp, fn, delay


# --------------------------------------------------------------------------- #
# Общий протокол 06/06b (единая сетка и сплит — конец расхождениям rc0)
# --------------------------------------------------------------------------- #

FAR_BUDGET = 2.0      # тревог на 1000 МО-мес (пересмотр 1.2 rc0 — см. FIX_A11_report)
N_MEAS = 400
SEED_SUB = 42         # выбор измерительных рядов
SEED_PLAN = 123       # план инъекций (общий для всех методов → парность)

GRIDS = {
    "PELT": ("pen", det_pelt, [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 14.0]),
    "BinSeg": ("pen", det_binseg, [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 14.0]),
    "Window": ("pen", det_window, [1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 20.0]),
    "CUSUM": ("h", det_cusum, [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]),
    "EWMA": ("L", det_ewma, [2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]),
    "BOCPD": ("thr", None, [0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 0.95]),
}
LEVEL_DELTAS = [0.4, 0.8, 1.6, 3.0, -0.8, -1.6]   # двусторонне (A05 P1-7)
TREND_DELTAS = [0.04, 0.08]
VAR_KAPPAS = [1.5, 2.0]
CELLS = ([("level", d) for d in LEVEL_DELTAS] +
         [("trend", d) for d in TREND_DELTAS] +
         [("variance", k) for k in VAR_KAPPAS])


def load_wide(root):
    """Панель → (wide DataFrame log_value 24 мес, категории рядов)."""
    import pandas as pd
    panel = pd.read_parquet(root / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    wide = panel.pivot_table(index="series_id", columns="month_idx",
                             values="log_value").reindex(columns=range(24))
    cats = np.array([s.split("|")[1] for s in wide.index])
    return wide, cats


def stratified_measurement_idx(wide_mat, cats, n=N_MEAS, seed=SEED_SUB):
    """Страта = категория × квартиль медианного уровня (R04 §2.1; в rc0 стратификации
    не было — rng.choice напрямую)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    med = np.nanmedian(wide_mat, axis=1)
    q = pd.qcut(pd.Series(med), 4, labels=False)
    strata = (pd.Series(cats) + "|" + pd.Series(q).astype(str)).to_numpy()
    idx = []
    per = max(1, n // np.unique(strata).size)
    for s in np.unique(strata):
        pool = np.where(strata == s)[0]
        idx.extend(rng.choice(pool, size=min(per, len(pool)), replace=False))
    return np.array(sorted(set(idx)))


def injection_plans(n, seed=SEED_PLAN):
    """tau на ряд для каждой ячейки (kind, delta) — один план на все методы."""
    rngp = np.random.default_rng(seed)
    return {cell: rngp.integers(8, 17, size=n) for cell in CELLS}


def detect_all(z, p_restart, method, param):
    """Единая точка входа: BOCPD использует предрасчёт p_restart, остальные — GRIDS."""
    if method == "BOCPD":
        return [b for b in range(1, len(z)) if p_restart[b] > param]
    return GRIDS[method][1](z, param)


def working_points(cal_df, budget=FAR_BUDGET):
    """Рабочая точка метода = самый чувствительный порог в бюджете (макс. FAR из
    допустимых). None — методу негде работать в бюджете (честная фиксация)."""
    frozen = {}
    for name in cal_df.method.unique():
        ok = cal_df[(cal_df.method == name) & (cal_df.far_per_1000 <= budget)]
        frozen[name] = float(ok.loc[ok.far_per_1000.idxmax()].param) if len(ok) else None
    return frozen
