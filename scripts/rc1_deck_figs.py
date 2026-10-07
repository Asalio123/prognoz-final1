"""rc1: дек-версии фигур для презентации (стиль = палитра/шрифты деки).

F3 2-панели (Богородский + Дальнереченский) — витрина детекции, без мелких
matplotlib-заголовков (заголовки живут в HTML деки). Выход: reports/figures/deck_F3.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]

# шрифт деки
fp = "/tmp/fonts/GolosText.ttf"
if Path(fp).exists():
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = "Golos Text"

# тёмная айдентика деки (СберИндекс): bg #0E1511, ink #F2F7F3
BG, INK, MUTED, GREEN, COPPER = "#0E1511", "#F2F7F3", "#93A697", "#3ECF6A", "#D8A24A"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": (236/255, 244/255, 238/255, 0.08), "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
})

panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
cp = pd.read_parquet(ROOT / "outputs" / "cpd_breaks_v2.parquet")

MONTHS = pd.period_range("2023-01", "2024-12", freq="M").astype(str).tolist()

cases = [
    (1267, "Маркетплейсы", "Богородский м.о., Кировская обл. · Маркетплейсы"),
    (788, "Общественное питание", "Дальнереченский м.р., Приморье · Общепит"),
]

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), dpi=200)
for ax, (tid, cat, _) in zip(axes, cases):
    s = panel[(panel["territory_id"].astype(str) == str(tid)) & (panel["category"] == cat)].sort_values("month_idx")
    x = np.arange(len(s))
    ax.plot(x, s["value"], color=INK, lw=2.0, solid_capstyle="round")
    br = cp[(cp["territory_id"].astype(str) == str(tid)) & (cp["category"] == cat)]
    for _, r in br.iterrows():
        c = GREEN if r["method"] in ("PELT", "BinSeg") else COPPER
        ax.axvline(int(r["break_month"]), color=c, lw=1.3, ls=(0, (4, 3)), alpha=0.9)
    for m in (0, 12):
        ax.axvline(m, color=MUTED, lw=0.6, alpha=0.35)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["2023-01", "2023-07", "2024-01", "2024-07", "2024-12"])
    ax.set_ylabel("₽/жителя")
    ax.margins(x=0.02)

fig.tight_layout(pad=0.8)
out = ROOT / "reports" / "figures" / "deck_F3.png"
fig.savefig(out, bbox_inches="tight", facecolor=BG)
print("saved", out)
