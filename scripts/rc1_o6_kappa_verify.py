"""rc1 / O6: независимая верификация macro-Cohen's κ=0.926 (outputs/news_kappa_v2.md).

Пересчёт из первичных данных: outputs/news_kappa_sample_v2.csv (50 заголовков) +
ручная разметка human (дословно перенесена из scripts/17_news_kappa.py, единственный
источник human-меток) + таксономия v3 (configs/news_taxonomy.yaml).
Сверяем: per-theme TP/FP/FN/TN, per-theme κ, macro-κ, micro P/R.

Независимость: своя реализация стемминга/κ (не импорт 17_news_kappa.py).
Выход: outputs/rc1_o6_kappa_verify.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ["prices_inflation", "fx_ruble", "rate_credit",
          "sanctions_shocks", "consumer_sentiment", "emergencies_regional"]
SHORT = {"prices_inflation": "prices", "fx_ruble": "fx", "rate_credit": "rate",
         "sanctions_shocks": "sanctions", "consumer_sentiment": "sentiment",
         "emergencies_regional": "emerg"}
HUMAN = {
    0: "", 1: "", 2: "prices,rate,fx", 3: "sanctions", 4: "emerg", 5: "sanctions",
    6: "prices", 7: "", 8: "prices", 9: "sanctions", 10: "emerg", 11: "prices",
    12: "sentiment", 13: "prices", 14: "sanctions", 15: "prices", 16: "", 17: "fx",
    18: "", 19: "emerg", 20: "", 21: "fx", 22: "emerg", 23: "rate", 24: "",
    25: "emerg", 26: "", 27: "sanctions", 28: "sanctions", 29: "sanctions",
    30: "prices", 31: "rate", 32: "rate", 33: "", 34: "", 35: "rate", 36: "rate",
    37: "prices", 38: "", 39: "emerg", 40: "emerg", 41: "sanctions", 42: "",
    43: "rate", 44: "rate", 45: "sanctions", 46: "", 47: "", 48: "rate", 49: "fx",
}
# Опубликованные значения из outputs/news_kappa_v2.md (сверочная цель)
PUBLISHED = {  # тема: (kappa, TP, FP, FN, TN)
    "prices_inflation": (0.929, 8, 1, 0, 41),
    "fx_ruble": (0.000, 0, 0, 4, 46),
    "rate_credit": (1.000, 9, 0, 0, 41),
    "sanctions_shocks": (0.852, 7, 0, 2, 41),
    "consumer_sentiment": (0.000, 0, 0, 1, 49),
    "emergencies_regional": (0.922, 7, 1, 0, 42),
}
PUBLISHED_MACRO = 0.926


def stem_regex(kw: str) -> str:
    parts = []
    for tok in kw.lower().split():
        tok = tok[:-1] if tok.endswith("*") else tok
        parts.append(re.escape(tok) + r"\w*")
    return r"\s+".join(parts)


def cohen_kappa_binary(d: np.ndarray, h: np.ndarray) -> float:
    po = np.mean(d == h)
    pe = d.mean() * h.mean() + (1 - d.mean()) * (1 - h.mean())
    return float("nan") if pe >= 1 else float((po - pe) / (1 - pe))


def main():
    tax = yaml.safe_load(open(ROOT / "configs" / "news_taxonomy.yaml"))
    geo_excl = re.compile(r"(?:^|\W)(?:" + "|".join(stem_regex(s) for s in tax["geo_exclude"]) + r")")
    geo_resc = re.compile(r"(?:^|\W)(?:" + "|".join(stem_regex(s) for s in tax["geo_rescue"]) + r")")

    s = pd.read_csv(ROOT / "outputs" / "news_kappa_sample_v2.csv")
    assert len(s) == 50 and set(s["id"]) == set(range(50))
    tl = s["title"].str.lower()
    is_f = tl.str.contains(geo_excl, regex=True)
    has_r = tl.str.contains(geo_resc, regex=True)

    lines = ["# Верификация κ=0.926 (rc1/O6) — пересчёт из news_kappa_sample_v2.csv", "",
             "| тема | κ (пересчёт) | κ (публ.) | Δ | TP | FP | FN | TN | статус |",
             "|---|---|---|---|---|---|---|---|---|"]
    active = [t for t in TOPICS if tax["topics"][t].get("status") != "retired"]
    kappas, TP, FP, FN = {}, 0, 0, 0
    ok_all = True
    for t in TOPICS:
        spec = tax["topics"][t]
        pat = re.compile(r"(?:^|\W)(?:" + "|".join(stem_regex(k) for k in spec["keywords"]) + r")")
        raw = tl.str.contains(pat, regex=True)
        if spec.get("status") == "retired":
            d = pd.Series(False, index=s.index)
        elif spec.get("geo") == "with_rescue":
            d = raw & ~(is_f & ~has_r)
        else:
            d = raw & ~is_f
        d = d.to_numpy()
        h = np.array([SHORT[t] in (HUMAN[i].split(",") if HUMAN[i] else []) for i in s["id"]])
        tp, fp = int((d & h).sum()), int((d & ~h).sum())
        fn, tn = int((~d & h).sum()), int((~d & ~h).sum())
        k = cohen_kappa_binary(d, h)
        kappas[t] = k
        pub_k, pub_tp, pub_fp, pub_fn, pub_tn = PUBLISHED[t]
        match = (abs(k - pub_k) < 5e-4 and (tp, fp, fn, tn) == (pub_tp, pub_fp, pub_fn, pub_tn))
        ok_all &= match
        if t in active:
            TP += tp; FP += fp; FN += fn
        lines.append(f"| {t}{' (retired)' if t not in active else ''} | {k:.3f} | {pub_k:.3f} | "
                     f"{k - pub_k:+.4f} | {tp} | {fp} | {fn} | {tn} | {'OK' if match else 'РАСХОЖДЕНИЕ'} |")

    macro = float(np.mean([kappas[t] for t in active]))
    prec, rec = TP / (TP + FP), TP / (TP + FN)
    lines += ["",
              f"macro-κ (пересчёт, активные {active}): **{macro:.6f}** → округление {macro:.3f} "
              f"vs публ. {PUBLISHED_MACRO} — {'СОВПАЛО' if abs(macro - PUBLISHED_MACRO) < 5e-4 else 'РАСХОДИТСЯ'}",
              f"micro precision: {TP}/{TP + FP} = {prec:.4f} (публ. 31/33 = 93.9%)",
              f"micro recall: {TP}/{TP + FN} = {rec:.4f} (публ. 31/33 = 93.9%)",
              "",
              f"Сводная сверка per-theme ячеек: {'все совпали' if ok_all else 'ЕСТЬ РАСХОЖДЕНИЯ'}.",
              "",
              "Оговорка о независимости: human-разметка существует только как литерал HUMAN в",
              "scripts/17_news_kappa.py (разметил разработчик); верификация воспроизводит вычисление",
              "κ из тех же входов своим кодом, но не является независимой переразметкой сэмпла."]
    out = "\n".join(lines)
    (ROOT / "outputs" / "rc1_o6_kappa_verify.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
