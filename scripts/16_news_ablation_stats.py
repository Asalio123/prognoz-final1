"""Статистика абляции новостного слоя (К5) — закрывает аудит rc0 P0-3 / A03 P1-5.

БЫЛО: только pooled ΔMAE% (−6.1% при h=1) без единого теста; «h=3 нейтрально» при +1.64%.
СТАЛО:
1) Парный Wilcoxon LGBM vs LGBM+news для h∈{1,3,6}:
   - первичная единица — origin (mean |err| по панели внутри origin'а; n=12/10/7 —
     честная временная выборка, pooled n=140k — иллюзия точности);
   - вторичная — ряд МО (mean |err| по origin'ам внутри series_id; n=2016, но строки
     внутри origin'а коррелированы — антиконсервативно, помечено);
   - Holm-коррекция по семейству из 3 горизонтов (первичный уровень).
2) Per-origin профиль ΔMAE с пометкой шоковых месяцев (гипотеза Andreou et al. 2013:
   прирост концентрирован в шоковые режимы) + Mann-Whitney shock vs calm.
3) Таблица ΔMAE% по 6 категориям × h — включая вредные (аудит: Общепит +34%,
   Транспорт +12% при h=1 в отчётной выборке не упоминались).

Шоковые месяцы-цели из research/C3_russia_shocks_2023_24.md:
ставка 21.07.2023; девальвация+внеплановая ставка 08.2023; топливо+ставка 09.2023;
ставка 15% 10.2023; яйца+ставка 16% 12.2023; паводок Орск 04.2024; санкции против
Мосбиржи 06.2024; ставка 21% 10.2024; девальвация 11.2024.
Маппинг: origin o (конец месяца o, 0-based от 2023-01) → цель месяц o+h−1
(проверено: origin 23, h=1 → декабрьский пик y_true 2024-12).

Вход: outputs/preds_lgbm.parquet, outputs/preds_lgbm_news.parquet (фичи таксономии v2 —
перегенерация на v3 и перезапуск LGBM — задача главного потока; тесты здесь остаются валидны,
т.к. считаются по паркетам предсказаний).
Выход: outputs/news_ablation_stats_v2.csv, outputs/news_ablation_by_origin_v2.csv,
       outputs/news_ablation_by_category.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
MONTHS = pd.period_range("2023-01", "2024-12", freq="M").astype(str)
SHOCK_MONTHS = {"2023-07", "2023-08", "2023-09", "2023-10", "2023-12",
                "2024-04", "2024-06", "2024-10", "2024-11"}


def holm(pvals):
    """Holm-adjusted p-values, сохраняет порядок входа."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    base = pd.read_parquet(ROOT / "outputs" / "preds_lgbm.parquet")
    news = pd.read_parquet(ROOT / "outputs" / "preds_lgbm_news.parquet")
    key = ["series_id", "origin", "horizon"]
    df = base.merge(news, on=key, suffixes=("_base", "_news"))
    assert (df["y_true_base"] == df["y_true_news"]).all(), "y_true различаются — merge сломан"
    df["ae_base"] = (df["y_true_base"] - df["y_pred_base"]).abs()
    df["ae_news"] = (df["y_true_base"] - df["y_pred_news"]).abs()
    df["category"] = df["series_id"].str.split("|").str[1]
    print(f"merged {len(df)} строк (base {len(base)}, news {len(news)})")

    stat_rows, origin_rows = [], []
    for h in [1, 3, 6]:
        sub = df[df.horizon == h]
        # origin-агрегат (первичная единица)
        o = sub.groupby("origin")[["ae_base", "ae_news"]].mean()
        o["d"] = o["ae_news"] - o["ae_base"]  # <0 = новости помогают
        w_o = wilcoxon(o["ae_base"], o["ae_news"])
        # series-агрегат (вторичная единица, антиконсервативно)
        s = sub.groupby("series_id")[["ae_base", "ae_news"]].mean()
        w_s = wilcoxon(s["ae_base"], s["ae_news"])
        stat_rows.append(dict(
            h=h, n_rows=len(sub), n_origins=len(o), n_series=len(s),
            mae_base=sub["ae_base"].mean(), mae_news=sub["ae_news"].mean(),
            delta_pct=100 * (sub["ae_news"].mean() / sub["ae_base"].mean() - 1),
            origin_mean_delta=o["d"].mean(), origins_improved=int((o["d"] < 0).sum()),
            wilcoxon_origin_p=w_o.pvalue, wilcoxon_series_p=w_s.pvalue))
        # per-origin профиль с шоковыми месяцами
        for orig, r in o.iterrows():
            tm = MONTHS[int(orig) + h - 1] if int(orig) + h - 1 < len(MONTHS) else None
            origin_rows.append(dict(h=h, origin=int(orig), target_month=tm,
                                    shock=tm in SHOCK_MONTHS if tm else None,
                                    mae_base=r["ae_base"], mae_news=r["ae_news"], delta_mae=r["d"]))
    st = pd.DataFrame(stat_rows)
    st["holm_p_origin"] = holm(st["wilcoxon_origin_p"].values)
    st.to_csv(ROOT / "outputs" / "news_ablation_stats_v2.csv", index=False)
    print("\n=== абляция LGBM vs LGBM+news (первичный тест: origin-level Wilcoxon) ===")
    print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    og = pd.DataFrame(origin_rows)
    og.to_csv(ROOT / "outputs" / "news_ablation_by_origin_v2.csv", index=False)
    print("\n=== per-origin ΔMAE (руб; <0 — новости помогают), * = шоковый месяц цели ===")
    for h in [1, 3, 6]:
        o = og[og.h == h]
        line = "  ".join(f"{int(r.origin)}{'*' if r.shock else ''}:{r.delta_mae:+.0f}" for r in o.itertuples())
        print(f"h={h}: {line}")
        sh, ca = o[o.shock]["delta_mae"], o[~o.shock]["delta_mae"]
        if len(sh) and len(ca):
            u = mannwhitneyu(sh, ca, alternative="less")  # H1: на шоках Δ меньше (помогают)
            print(f"      shock n={len(sh)} mean Δ={sh.mean():+.1f} | calm n={len(ca)} mean Δ={ca.mean():+.1f} "
                  f"| MW p(шок лучше)={u.pvalue:.4f}")

    # таблица по категориям × h (включая вредные — аудит P0-3)
    cat = df.groupby(["horizon", "category"])[["ae_base", "ae_news"]].mean().reset_index()
    cat["delta_pct"] = 100 * (cat["ae_news"] / cat["ae_base"] - 1)
    cat = cat.rename(columns={"horizon": "h", "ae_base": "mae_base", "ae_news": "mae_news"})
    cat.to_csv(ROOT / "outputs" / "news_ablation_by_category.csv", index=False)
    print("\n=== ΔMAE% по категориям × h ===")
    print(cat.pivot(index="category", columns="h", values="delta_pct").round(2).to_string())


if __name__ == "__main__":
    main()
