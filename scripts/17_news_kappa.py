"""κ-валидация словаря (К5) — честная замена несостоявшейся rubert-tiny2 валидации
(аудит rc0 P0-2: «валидация rubert-tiny2 по Cohen's kappa» в шапке v2 была декларацией
без кода и числа). Протокол: стратифицированный сэмпл n=50 (8 на активную тему v3,
6 из retired-тем, 12 непокрытых; seed 11), ручная разметка разработчиком (human)
по схеме всех 6 тем с критерием «тема релевантна каналу РФ-потребления»
(иностранное событие -> без темы), затем Cohen's κ словарь-vs-человек по каждой теме.
Сэмпл был вытянут по меткам v3 до расширения geo_exclude двойными формами стран —
итоговые метки словаря пересчитаны на том же сэмпле финальной таксономией.

Выход: outputs/news_kappa_v2.md
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

# Ручная разметка (human) по id сэмпла — размечал разработчик 06.10.2026, критерий выше.
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
# Журнал разногласий (id: комментарий)
NOTES = {
    1: "dict FP (emerg): пожар в Познани — geo-список работает на уровне названий стран, город без маркера страны ускользает (известное ограничение)",
    12: "dict=prices vs human=sentiment: «инфляционные ожидания россиян» — канонический sentiment (тема retired); в prices попало по стему «инфляц» — спорно",
    17: "human fx, dict none: «оптимальный курс рубля» — цена retirement'а fx_ruble (объём)",
    21: "human fx, dict none: «реальный эффективный курс рубля вырос» — то же",
    26: "до расширения geo-списка был dict FP (санкции ЕС против Молдавии); стем «молдав» добавлен — исправлено",
    29: "human sanctions, dict none: «потолок цен на нефтепродукты из РФ» — recall-пропуск словаря (нет лексемы «потолок цен»)",
    41: "human sanctions, dict none: «Чехия откажется от российской нефти» — recall-пропуск (нет лексемы отказ/импортозамещение); пограничный кейс",
    49: "human fx, dict none: «укрепление рубля» — цена retirement'а fx",
}


def stem_regex(kw: str) -> str:
    parts = []
    for tok in kw.lower().split():
        if tok.endswith("*"):
            parts.append(re.escape(tok[:-1]) + r"\w*")
        else:
            parts.append(re.escape(tok) + r"\w*")
    return r"\s+".join(parts)


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = (a.mean() * b.mean()) + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def main():
    tax = yaml.safe_load(open(ROOT / "configs" / "news_taxonomy.yaml"))
    geo_excl = re.compile(r"(?:^|\W)(" + "|".join(stem_regex(s) for s in tax["geo_exclude"]) + r")")
    geo_resc = re.compile(r"(?:^|\W)(" + "|".join(stem_regex(s) for s in tax["geo_rescue"]) + r")")
    kw_pats = {t: re.compile(r"(?:^|\W)(" + "|".join(stem_regex(k) for k in spec["keywords"]) + r")")
               for t, spec in tax["topics"].items()}

    s = pd.read_csv(ROOT / "outputs" / "news_kappa_sample_v2.csv")
    s["t"] = s["title"].str.lower()
    is_f = s["t"].str.contains(geo_excl)
    has_r = s["t"].str.contains(geo_resc)

    dict_lab, hum_lab = {}, {}
    for t in TOPICS:
        spec = tax["topics"][t]
        raw = s["t"].str.contains(kw_pats[t])
        if spec.get("status") == "retired":
            d = pd.Series(False, index=s.index)
        elif spec.get("geo") == "with_rescue":
            d = raw & ~(is_f & ~has_r)
        else:
            d = raw & ~is_f
        dict_lab[t] = d.to_numpy()
        hum_lab[t] = np.array([SHORT[t] in (HUMAN[i].split(",") if HUMAN[i] else []) for i in s["id"]])

    lines = ["# κ-валидация словаря v3 против ручной разметки (n=50)",
             "",
             "Замена заявленной и не проведённой rubert-tiny2 валидации (аудит P0-2).",
             "Протокол: стратифицированный сэмпл (8/активную тему, 6 retired, 12 непокрытых, seed 11);",
             "human = ручная разметка разработчиком по критерию «тема релевантна каналу РФ-потребления»;",
             "dict = таксономия v3 (retired-темы не метят — их κ = цена retirement'а).",
             "",
             "| тема | κ | TP | FP | FN | TN |", "|---|---|---|---|---|---|"]
    active = [t for t in TOPICS if tax["topics"][t].get("status") != "retired"]
    kappas = {}
    for t in TOPICS:
        d, h = dict_lab[t], hum_lab[t]
        tp = int((d & h).sum()); fp = int((d & ~h).sum()); fn = int((~d & h).sum()); tn = int((~d & ~h).sum())
        k = kappa(d, h)
        kappas[t] = k
        lines.append(f"| {t}{' (retired)' if t not in active else ''} | {k:.3f} | {tp} | {fp} | {fn} | {tn} |")
    macro = float(np.mean([kappas[t] for t in active]))
    TP = sum(int((dict_lab[t] & hum_lab[t]).sum()) for t in active)
    FP = sum(int((dict_lab[t] & ~hum_lab[t]).sum()) for t in active)
    FN = sum(int((~dict_lab[t] & hum_lab[t]).sum()) for t in active)
    lines += ["",
              f"macro-κ по активным темам: **{macro:.3f}**",
              f"micro precision (активные): {TP}/{TP + FP} = {TP / (TP + FP):.1%}; "
              f"micro recall: {TP}/{TP + FN} = {TP / (TP + FN):.1%}",
              "",
              "Контекст: аудит rc0 для v2 оценивал взвешенный precision ≈72% (A12) / ≈85% по каналу (A03).",
              "",
              "## Журнал разногласий"]
    for i, note in NOTES.items():
        row = s[s.id == i].iloc[0]
        lines.append(f"- [{row['date']}] «{row['title']}» — {note}")
    (ROOT / "outputs" / "news_kappa_v2.md").write_text("\n".join(lines))
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
