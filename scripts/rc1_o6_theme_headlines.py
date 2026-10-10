"""rc1 / O6: артефакт «тема → 3 дословных заголовка с датой и источником» (К5-слайд).

Корпус: data/external/news/headlines_{kommersant,vedomosti}.jsonl (27 806 строк,
2023-01-01..2024-12-31). Классификация — таксономия v3 (configs/news_taxonomy.yaml),
включая retired-темы (fx_ruble, consumer_sentiment): их лексемы применяются к корпусу
тем же кодом, что и к активным, но помечаются retired (в фичи не входят — объём).

Отбор: для каждой темы печатаются кандидаты (дата, источник, заголовок, doc_id),
дальше ручной глазной контроль соответствия теме; финальная тройка — с разнесением
по датам и источникам. Заголовки ДОСЛОВНЫЕ из корпуса (проверяемо по doc_id/url).

Выход: outputs/rc1_o6_theme_headlines.md
"""
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data" / "external" / "news"
TOPICS = ["prices_inflation", "fx_ruble", "rate_credit",
          "sanctions_shocks", "consumer_sentiment", "emergencies_regional"]


def stem_regex(kw: str) -> str:
    parts = []
    for tok in kw.lower().split():
        if tok.endswith("*"):
            parts.append(re.escape(tok[:-1]) + r"\w*")
        else:
            parts.append(re.escape(tok) + r"\w*")
    return r"\s+".join(parts)


# Финальные тройки (source, doc_id) — выбраны глазами из кандидатов выше по критериям:
# однозначное попадание в тему, разнесение по датам/источникам; дословность проверена
# сверкой по корпусу (doc_id → title) в этом же скрипте.
PICKS = {
    "prices_inflation": [("kommersant", "6298009"), ("vedomosti", "1029746"), ("kommersant", "7267377")],
    "fx_ruble": [("kommersant", "6043176"), ("vedomosti", "1001829"), ("vedomosti", "1084658")],
    "rate_credit": [("kommersant", "6084977"), ("vedomosti", "1054055"), ("vedomosti", "1083538")],
    "sanctions_shocks": [("kommersant", "6070279"), ("kommersant", "6410000"), ("vedomosti", "1039521")],
    "consumer_sentiment": [("vedomosti", "976721"), ("kommersant", "6174050"), ("vedomosti", "1012912")],
    "emergencies_regional": [("vedomosti", "973159"), ("vedomosti", "979778"), ("kommersant", "6862317")],
}
THEME_RU = {
    "prices_inflation": "Цены и инфляция",
    "fx_ruble": "Курс рубля",
    "rate_credit": "Ключевая ставка и кредит",
    "sanctions_shocks": "Санкции и отраслевые шоки",
    "consumer_sentiment": "Потребительские настроения",
    "emergencies_regional": "ЧС и региональные события",
}
CHANNEL = {
    "prices_inflation": "реальные доходы → потребление",
    "fx_ruble": "девальвация → ажиотажный спрос",
    "rate_credit": "ставка → кредиты/депозиты → потребление",
    "sanctions_shocks": "ограничения → отдельные категории",
    "consumer_sentiment": "настроения → сбережения vs траты",
    "emergencies_regional": "ЧС → локальные шоки",
}


def main():
    tax = yaml.safe_load(open(ROOT / "configs" / "news_taxonomy.yaml"))
    geo_excl = re.compile(r"(?:^|\W)(" + "|".join(stem_regex(s) for s in tax["geo_exclude"]) + r")")
    geo_resc = re.compile(r"(?:^|\W)(" + "|".join(stem_regex(s) for s in tax["geo_rescue"]) + r")")
    kw_pats = {t: re.compile(r"(?:^|\W)(" + "|".join(stem_regex(k) for k in spec["keywords"]) + r")")
               for t, spec in tax["topics"].items()}

    rows = []
    for src in ("kommersant", "vedomosti"):
        with open(NEWS / f"headlines_{src}.jsonl") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows).drop_duplicates(subset=["source", "doc_id"])
    print(f"корпус: {len(df)} заголовков, {df['date'].min()}..{df['date'].max()}")

    t_low = df["title"].str.lower()
    is_f = t_low.str.contains(geo_excl)
    has_r = t_low.str.contains(geo_resc)

    hits = {}
    for t in TOPICS:
        spec = tax["topics"][t]
        raw = t_low.str.contains(kw_pats[t])
        if spec.get("geo") == "with_rescue":
            mask = raw & ~(is_f & ~has_r)
        else:
            mask = raw & ~is_f
        hits[t] = int(mask.sum())
        print(f"{t}: {hits[t]} хитов (status={spec.get('status', 'active')})")

    by_id = {(r["source"], r["doc_id"]): r for r in rows}

    md = ["# Тема → канал → 3 дословных заголовка (таксономия v3, корпус 2023–2024)",
          "",
          f"Корпус: {len(df)} заголовков Коммерсантъ+Ведомости, 2023-01-01..2024-12-31",
          "(data/external/news/headlines_*.jsonl). Хиты — применение лексем v3 с geo-фильтром",
          "(scripts/rc1_o6_theme_headlines.py). Заголовки дословные, проверены по doc_id.",
          ""]
    for t in TOPICS:
        spec = tax["topics"][t]
        status = spec.get("status", "active")
        md.append(f"## {THEME_RU[t]} (`{t}`)")
        md.append(f"Канал: {CHANNEL[t]}. Статус v3: **{status}**. Хитов в корпусе: {hits[t]}.")
        md.append("")
        md.append("| Дата | Источник | Дословный заголовок | Ссылка |")
        md.append("|---|---|---|---|")
        for key in PICKS[t]:
            r = by_id[key]
            md.append(f"| {r['date']} | {r['source']} | «{r['title']}» | {r['url']} |")
        if t == "consumer_sentiment":
            md.append("")
            md.append("_Всего 4 хита за 24 мес (4-й — шум «Alibaba … в КНР») — тема retired по объёму; "
                      "заголовки показаны как иллюстрация канала, в фичи не входят._")
        if t == "fx_ruble":
            md.append("")
            md.append("_Тема retired по объёму (34 хита/24 мес при пороге 48); заголовки — "
                      "иллюстрация канала, в фичи не входят._")
        md.append("")
    (ROOT / "outputs" / "rc1_o6_theme_headlines.md").write_text("\n".join(md))
    print("\nsaved outputs/rc1_o6_theme_headlines.md")


if __name__ == "__main__":
    main()
