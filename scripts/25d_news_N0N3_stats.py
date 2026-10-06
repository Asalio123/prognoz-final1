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


def load(name):
    df = pd.read_parquet(ROOT / "outputs" / name)
    df["ae"] = (df["y_true"] - df["y_pred"]).abs()
    return df


def main():
    n0 = load("preds_lgbm.parquet")[["series_id", "origin", "horizon", "ae"]].rename(columns={"ae": "ae_n0"})
    rows = []
    for cfg, fname in CONFIGS.items():
        nk = load(fname)[["series_id", "origin", "horizon", "ae"]].rename(columns={"ae": "ae_cfg"})
        df = n0.merge(nk, on=["series_id", "origin", "horizon"], how="inner")
        for h in [1, 3, 6]:
            sub = df[df.horizon == h]
            o = sub.groupby("origin")[["ae_n0", "ae_cfg"]].mean()
            w = wilcoxon(o["ae_n0"], o["ae_cfg"])
            ea = sub.set_index(["origin", "series_id"])["ae_cfg"]
            eb = sub.set_index(["origin", "series_id"])["ae_n0"]
            dm = dm_test(ea, eb, h)  # mean_diff < 0 = конфиг лучше N0
            mae0, maec = sub["ae_n0"].mean(), sub["ae_cfg"].mean()
            rows.append(dict(config=cfg, h=h, n_origins=len(o), n_rows=len(sub),
                             mae_n0=mae0, mae_cfg=maec,
                             relmae=maec / mae0,
                             delta_pct=100 * (maec / mae0 - 1),
                             origin_mean_delta=(o["ae_cfg"] - o["ae_n0"]).mean(),
                             origins_improved=int(((o["ae_cfg"] - o["ae_n0"]) < 0).sum()),
                             wilcoxon_origin_p=w.pvalue,
                             dm_hln_p=dm["p"]))
            print(f"{cfg} h={h}: RelMAE={maec/mae0:.4f} Δ={100*(maec/mae0-1):+.2f}% "
                  f"Wilcoxon p={w.pvalue:.4f} DM p={dm['p']:.4f} "
                  f"origins улучш. {int(((o['ae_cfg']-o['ae_n0'])<0).sum())}/{len(o)}", flush=True)

    st = pd.DataFrame(rows)
    # Holm внутри семейства {N1,N2,N3} per горизонт (первичный тест = Wilcoxon origin)
    st["holm_p"] = np.nan
    for h in [1, 3, 6]:
        sub = st[st.h == h]
        adj = holm(dict(zip(sub["config"], sub["wilcoxon_origin_p"])))
        st.loc[st.h == h, "holm_p"] = st.loc[st.h == h, "config"].map(adj)
    st["decision_rule_pass"] = ((st.h == 1) & (st.relmae < 1)) | (st.holm_p < 0.10)
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
