"""rc1 S4: исправленная F7 — профиль старения прогноза.

Проблемы оригинала (reports/figures/F7_step_profile.png):
- 9 линий, жёлтый #F0E442 (chronos2) почти невиден на белом и пропадает в грейскейле;
- легенда 8pt в 2 колонки наезжает на данные (LightGBM h=1 = 2649 упирается в неё);
- заголовок-тема, а не вывод; мелкие кегли для слайда.

Фикс: 6 показательных моделей, палитра Okabe-Ito без жёлтого, прямые подписи
у концов линий вместо легенды, заголовок-вывод, dpi=200.
Данные и агрегация — дословно как в scripts/17_figures.py (f7_step_profile).
Запуск: .venv/bin/python scripts/rc1_s4_fix_f7.py
Выход: outputs/rc1_s4_F7_step_profile.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

plt.rcParams.update({
    "font.family": "PT Sans", "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.color": "#888888",
    "font.size": 12.5, "axes.labelsize": 13, "axes.titlesize": 14.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
})
# Okabe-Ito без жёлтого #F0E442 (невидим на белом / в грейскейле)
OK = {"Prophet-A": "#CC79A7", "AutoETS": "#D55E00", "HistoricAverage": "#E69F00",
      "LightGBM-direct": "#56B4E9", "chronos2_zs_log": "#009E73",
      "SeasonalNaive": "#000000"}
KEEP = list(OK.keys())


def load_pool():
    files = [f for f in OUT.glob("preds_*.parquet")
             if not f.match("*_ces_part*") and not f.match("*_steps_part*")
             and not f.match("*_quant_o12*") and not f.match("*_raw_o12*")
             and not f.match("*coldstart*")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df["model"] != "y"].drop_duplicates(["series_id", "origin", "horizon", "model"])


def main():
    df = load_pool()
    frames = []
    steps_files = list(OUT.glob("preds_stats_steps_part*.parquet"))
    if steps_files:
        s = pd.concat([pd.read_parquet(f) for f in steps_files], ignore_index=True)
        s["ae"] = (s.y_true - s.y_pred).abs()
        frames.append(s[["model", "horizon", "series_id", "ae"]])
    main_o12 = df[df.origin == 12].copy()
    main_o12["ae"] = (main_o12.y_true - main_o12.y_pred).abs()
    frames.append(main_o12[["model", "horizon", "series_id", "ae"]])
    allsteps = (pd.concat(frames, ignore_index=True)
                  .drop_duplicates(["model", "horizon", "series_id"]))
    prof = allsteps.groupby(["model", "horizon"])["ae"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for m in KEEP:
        p = prof[prof.model == m].sort_values("horizon")
        if p.empty:
            print("пропуск (нет данных):", m)
            continue
        if m == "SeasonalNaive":
            ax.plot(p.horizon, p.ae, marker="s", ms=5, lw=2.2, color=OK[m],
                    ls="--", zorder=5)
        else:
            ax.plot(p.horizon, p.ae, marker="o", ms=4.5, lw=2.0, color=OK[m])
        # прямая подпись: у конца линии; LightGBM обрывается на h=6 — подпись у пика h=1,
        # где вокруг пусто (иначе текст ложится на линию HistoricAverage)
        label = m.replace("_zs_log", "")
        if m == "LightGBM-direct":
            ax.annotate(label, (p.horizon.iloc[0], p.ae.iloc[0]),
                        xytext=(8, 8), textcoords="offset points", fontsize=11.5,
                        color=OK[m], va="bottom", fontweight="bold")
        else:
            ax.annotate(label, (p.horizon.iloc[-1], p.ae.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", fontsize=11.5,
                        color=OK[m], va="center", fontweight="bold")
    ax.set_xlim(0.6, 13.6)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Шаг прогноза (месяцев вперёд)")
    ax.set_ylabel("MAE, ₽/жителя")
    ax.set_title("Ошибка foundation-моделей к 12-му месяцу удваивается,\n"
                 "статистика и Prophet-A стареют мягко (train=2023, test=2024)")
    fig.tight_layout()
    out = OUT / "rc1_s4_F7_step_profile.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
