"""N2-агрегаты новостного слоя (К5): внутримесячное взвешивание mean/eom7/exp.

Вход: outputs/news_daily_v2.parquet (дневные счётчики 4 активных тем, таксономия v3),
      outputs/news_monthly_v2.parquet (mean-агрегаты + count + z — переиспользуются),
      data/external/news/headlines_*.jsonl (только для дневного знаменателя total —
      тот же корпус, та же загрузка, что в 11_run_news.py).

Агрегаты per тема k, месяц m (R07 §2.4, D3 §2.2):
- intensity_kd = count_k(d) / total(d) — дневная доля темы в ленте;
- mean  : среднее intensity по дням месяца        (= share_* из v2, не трогаем);
- eom7  : среднее intensity по последним 7 календарным дням месяца;
- exp   : взвешенное среднее intensity, w_d ∝ 2^(-Δd/7), Δd = дней до конца месяца;
          полураспад 7 дней, θ ПРЕДЗАДАН ex ante — не оценивается (D3 §2.3: T=24
          месячные точки не выносят оценку весовой функции).

Выход: outputs/news_monthly_v3.parquet = v2 целиком + 8 колонок
       (share_<topic>_eom7, share_<topic>_exp). v2-файл НЕ перезаписывается.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data" / "external" / "news"
HALF_LIFE_DAYS = 7.0  # θ предзадан (R07 §2.4 / D3 §2.3) — НЕ оценивать

TOPICS = ["prices_inflation", "rate_credit", "sanctions_shocks", "emergencies_regional"]


def load_daily_totals() -> pd.Series:
    """Общее число заголовков корпуса по дням (знаменатель intensity)."""
    rows = []
    for f in NEWS.glob("headlines_*.jsonl"):
        with open(f) as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.groupby("date").size().rename("total")


def main():
    daily = pd.read_parquet(ROOT / "outputs" / "news_daily_v2.parquet")
    daily.index = pd.to_datetime(daily.index).normalize()
    daily = daily[TOPICS]
    total = load_daily_totals()
    daily = daily.join(total, how="left")
    assert daily["total"].notna().all(), "в корпусе есть дни без totals — join сломан"
    print(f"дневной ряд: {daily.index.min().date()}..{daily.index.max().date()}, "
          f"{len(daily)} дней; total: min={daily['total'].min()}, "
          f"медиана={daily['total'].median():.0f}")

    # дневная intensity (доля темы в ленте дня); дней с total=0 в корпусе нет, но страхуемся
    inten = daily[TOPICS].div(daily["total"].replace(0, np.nan), axis=0)

    v2 = pd.read_parquet(ROOT / "outputs" / "news_monthly_v2.parquet")
    out = v2.copy()
    months = pd.period_range("2023-01", "2024-12", freq="M")
    assert list(v2.index.astype(str)) == list(months.astype(str)), "сетка месяцев v2 не 2023-01..2024-12"

    recs = []
    for m in months:
        days = pd.date_range(m.start_time, m.end_time, freq="D")
        sub = inten.reindex(days)  # все дни месяца покрыты корпусом (2023-01-01..2024-12-31)
        assert sub.notna().all().all(), f"пропуски в дневном ряду за {m}"
        # eom7: последние 7 календарных дней месяца
        eom7 = sub.iloc[-7:].mean()
        # exp: вес 2^(-Δd/7), Δd = дней до конца месяца (конец месяца = вес 1)
        delta = (m.end_time - days).days.to_numpy().astype(float)
        w = np.power(2.0, -delta / HALF_LIFE_DAYS)
        exp = (sub.to_numpy() * w[:, None]).sum(axis=0) / w.sum()
        rec = {"month": str(m)}
        for i, t in enumerate(TOPICS):
            rec[f"share_{t}_eom7"] = eom7[t]
            rec[f"share_{t}_exp"] = exp[i]
        recs.append(rec)
    add = pd.DataFrame(recs).set_index("month")
    add.index = pd.PeriodIndex(add.index, freq="M")
    out.index = pd.PeriodIndex(out.index.astype(str), freq="M")  # v2 писался со строковым индексом
    out = out.join(add)
    out.index.name = "month"
    out.to_parquet(ROOT / "outputs" / "news_monthly_v3.parquet")
    print(f"saved outputs/news_monthly_v3.parquet: {out.shape} "
          f"(v2: {v2.shape[1]} колонок + 8 новых)")
    # sanity: eom7/exp на известном шоке (ставка 21% — 25.10.2024, конец месяца)
    chk = out.loc["2024-10", ["share_rate_credit", "share_rate_credit_eom7", "share_rate_credit_exp"]]
    print("санити 2024-10 rate_credit: mean={:.4f} eom7={:.4f} exp={:.4f} "
          "(шок в конце месяца -> eom7/exp > mean)".format(*chk.values))
    assert chk["share_rate_credit_eom7"] > chk["share_rate_credit"], "eom7 должен усиливать конецмесячный шок"


if __name__ == "__main__":
    main()
