"""Сводная статистика абляции новостного слоя N0->N3 (К5; R07 §5 / D3 §6).

Конфиги (все — LightGBM-direct, сетка B3, сид 42, отличаются ТОЛЬКО новостным блоком):
  N0 = outputs/preds_lgbm.parquet           (без новостей)
  N1 = outputs/preds_lgbm_news_v2.parquet   (4 темы x {share, z_exp} mean-агрегаты)
  N2 = outputs/preds_lgbm_newsN2_v3.parquet (N1 + eom7/exp взвешивания, 25a/25c)
  N3 = outputs/preds_lgbm_newsN3_v3.parquet (N2 + 3 взаимодействия L_{i,k} x NF_k, 25c)

Протокол (как в 16_news_ablation_stats.py, расширен на семейство N1..N3):
- первичная единица — origin (mean |err| по панели внутри origin'а; n=12/10/7
  для h=1/3/6); pooled n=140k не используется как единица теста (иллюзия точности);
- парный Wilcoxon по origin'ам (N_k vs N0) — из scipy, как в 16; Holm-поправка
  внутри семейства {N1,N2,N3} per горизонт — src/significance.holm;
- дополнительно DM+HLN из src/significance.dm_test (вторичный тест, та же единица).

Предрегистрированное правило решения (decision.json / R07 §5 / D3 §6.2):
  RelMAE(N_k) < 1 на h=1  ИЛИ  p_Holm < 0.10  ->  слой в финальную модель.
  Иначе — слой НЕ входит, раздел остаётся полным (критерий оценивает способ).

Выход: outputs/news_ablation_N0N3.csv (конфиг x горизонт: ΔMAE%, RelMAE,
       Wilcoxon p, DM p, Holm p, вердикт по правилу).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.significance import dm_test, holm

CONFIGS = {
    "N1": "preds_lgbm_news_v2.parquet",
    "N2": "preds_lgbm_newsN2_v3.parquet",
    "N3": "preds_lgbm_newsN3_v3.parquet",
}
# парные сравнения соседних ступеней — прямые ответы на вопросы спеки:
# N2 vs N1 = «помогает ли внутримесячное взвешивание», N3 vs N2 = «помогает ли дифференциация»
PAIRWISE = [("N2", "N1"), ("N3", "N2")]


def load(name):
    df = pd.read_parquet(ROOT / "outputs" / name)
    df["ae"] = (df["y_true"] - df["y_pred"]).abs()
    return df


def compare(base_ae, cfg_ae, label_a, label_b, rows):
    """base_ae/cfg_ae: df[series_id, origin, horizon, ae]. Пишет строки per h в rows."""
    df = base_ae.merge(cfg_ae, on=["series_id", "origin", "horizon"], suffixes=("_a", "_b"))
    for h in [1, 3, 6]:
        sub = df[df.horizon == h]
        o = sub.groupby("origin")[["ae_a", "ae_b"]].mean()
        w = wilcoxon(o["ae_a"], o["ae_b"])
        ea = sub.set_index(["origin", "series_id"])["ae_b"]
        eb = sub.set_index(["origin", "series_id"])["ae_a"]
        dm = dm_test(ea, eb, h)  # mean_diff < 0 = B лучше A
        maea, maeb = sub["ae_a"].mean(), sub["ae_b"].mean()
        rows.append(dict(config=label_b, vs=label_a, h=h, n_origins=len(o), n_rows=len(sub),
                         mae_vs=maea, mae_cfg=maeb,
                         relmae=maeb / maea,
                         delta_pct=100 * (maeb / maea - 1),
                         origin_mean_delta=(o["ae_b"] - o["ae_a"]).mean(),
                         origins_improved=int(((o["ae_b"] - o["ae_a"]) < 0).sum()),
                         wilcoxon_origin_p=w.pvalue,
                         dm_hln_p=dm["p"]))
        print(f"{label_b} vs {label_a} h={h}: RelMAE={maeb/maea:.4f} Δ={100*(maeb/maea-1):+.2f}% "
              f"Wilcoxon p={w.pvalue:.4f} DM p={dm['p']:.4f} "
              f"origins улучш. {int(((o['ae_b']-o['ae_a'])<0).sum())}/{len(o)}", flush=True)


def main():
    n0 = load("preds_lgbm.parquet")[["series_id", "origin", "horizon", "ae"]]
    cfgs = {k: load(v)[["series_id", "origin", "horizon", "ae"]] for k, v in CONFIGS.items()}
    rows = []
    for cfg in CONFIGS:
        compare(n0, cfgs[cfg], "N0", cfg, rows)
    for a, b in PAIRWISE:
        compare(cfgs[a], cfgs[b], a, b, rows)

    st = pd.DataFrame(rows)
    # Holm внутри семейства {N1,N2,N3} vs N0 per горизонт (первичный тест = Wilcoxon origin);
    # парные ступенчатые сравнения — диагностические, в семейство Holm не входят
    st["holm_p"] = np.nan
    main_rows = st.vs == "N0"
    for h in [1, 3, 6]:
        mask = main_rows & (st.h == h)
        adj = holm(dict(zip(st.loc[mask, "config"], st.loc[mask, "wilcoxon_origin_p"])))
        st.loc[mask, "holm_p"] = st.loc[mask, "config"].map(adj)
    st["decision_rule_pass"] = main_rows & (((st.h == 1) & (st.relmae < 1)) | (st.holm_p < 0.10))
    st.to_csv(ROOT / "outputs" / "news_ablation_N0N3.csv", index=False)
    print(f"\nsaved outputs/news_ablation_N0N3.csv")
    print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== вердикт по предрегистрированному правилу (RelMAE<1 на h=1 ИЛИ p_Holm<0.10) ===")
    for cfg in CONFIGS:
        ok = st[(st.config == cfg) & st.decision_rule_pass]
        print(f"{cfg}: {'ПРОШЁЛ' if len(ok) else 'НЕ ПРОШЁЛ'}"
              + (f" (h={ok['h'].tolist()})" if len(ok) else ""))


if __name__ == "__main__":
    main()
