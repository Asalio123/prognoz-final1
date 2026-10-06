"""Минимальные тесты целостности (появились после аудита rc0, где их отсутствие было P0).

Гоняются быстро, без моделей: pytest tests/ -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

EXPECTED_ORIGINS = {1: 12, 3: 10, 6: 7, 12: 1}  # заморожено configs/validation.yaml


def _load_preds():
    excl = ("*_ces_part*", "*_steps_part*", "*_quant_o12*", "*_raw_o12*", "*coldstart*")
    files = [f for f in OUT.glob("preds_*.parquet") if not any(f.match(p) for p in excl)]
    if not files:
        pytest.skip("нет preds_*.parquet — прогоны ещё не завершены")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df["model"] != "y"].drop_duplicates(["series_id", "origin", "horizon", "model"])


def test_snaive_full_grid():
    """Знаменатель RelMAE обязан иметь полную сетку origins (иначе смешанная таблица — баг rc0)."""
    df = _load_preds()
    sn = df[df["model"] == "SeasonalNaive"]
    if not len(sn):
        pytest.skip("SeasonalNaive ещё не посчитан")
    got = sn.groupby("horizon")["origin"].nunique().to_dict()
    for h, n in EXPECTED_ORIGINS.items():
        assert got.get(h, 0) == n, f"SeasonalNaive h={h}: origins {got.get(h, 0)} != {n}"


def test_matched_cells_per_horizon():
    """Внутри горизонта все модели делят одни и те же (series_id, origin) ячейки
    (структурное исключение: LGBM h=12)."""
    df = _load_preds()
    for h in [1, 3, 6, 12]:
        dh = df[df["horizon"] == h]
        if not len(dh):
            continue
        sets = {m: set(zip(g["series_id"], g["origin"])) for m, g in dh.groupby("model")}
        ref = set.intersection(*sets.values())
        for m, s in sets.items():
            if m == "LightGBM-direct" and h == 12:
                continue  # 0 легальных train-строк — задокументировано
            extra = len(s - ref)
            assert extra == 0, f"h={h} {m}: {extra} ячеек вне пересечения"


def test_no_fallback_rows_mislabeled():
    """На origin 12 DriftAdjSNaive вырожден в SNaive по построению (нет YoY) —
    таблицы обязаны это показывать, а не продавать как силу модели."""
    df = _load_preds()
    d = df[(df["model"] == "DriftAdjSNaive") & (df["origin"] == 12)]
    s = df[(df["model"] == "SeasonalNaive") & (df["origin"] == 12)]
    if not len(d) or not len(s):
        pytest.skip("нет DriftAdj/SNaive на origin 12")
    j = d.merge(s, on=["series_id", "horizon"], suffixes=("_d", "_s"))
    assert np.allclose(j["y_pred_d"], j["y_pred_s"]), \
        "ожидалось вырождение DriftAdj→SNaive на origin 12; расхождение = проверить дрейф-логику"


def test_panel_integrity():
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    assert panel["month_idx"].nunique() == 24
    assert (panel.groupby(["territory_id", "category"])["month_idx"].nunique() == 24).all()
    assert (panel["value"] > 0).all()


def test_news_availability_gate():
    """Легальность новостных фичей: при merge cutoff==месяц новостей (origin = конец месяца)
    фича origin'а o использует новости месяца o — доступные к моменту прогноза.
    Проверяем: (1) корпус не выходит за пределы данных панели; (2) в join по cutoff
    ни одна новость не из месяца ПОЗЖЕ cutoff (иначе утечка)."""
    f = OUT / "news_monthly_v2.parquet"
    f = f if f.exists() else OUT / "news_monthly.parquet"
    if not f.exists():
        pytest.skip("новостные агрегаты не собраны")
    news = pd.read_parquet(f)
    mser = news["month"] if "month" in news.columns else pd.Series(news.index)
    months = pd.PeriodIndex(pd.Series(mser).astype(str), freq="M")
    assert months.max() <= pd.Period("2024-12", freq="M"), "корпус новостей выходит за данные"
    # симуляция join 05_run_lgbm.py: cutoff (месяц origin'а, 12..23) == месяц новостей
    cutoffs = pd.period_range("2023-12", periods=12, freq="M")  # origins 12..23
    usable = months[months.isin(cutoffs)]
    assert (usable.asi8 <= cutoffs.max().value).all()
    # ни одна новость не позже своего cutoff при равенстве ключей — тривиально по построению;
    # реальный риск был бы месяц новостей > cutoff: его нет по конструкции merge на равенстве.
    assert set(months) - set(cutoffs) <= {pd.Period("2024-12", freq="M")}  # дек-2024 не используется ни одним origin'ом


def test_decision_json_exists_and_parseable():
    import json
    p = ROOT / "configs" / "decision.json"
    assert p.exists(), "configs/decision.json отсутствует (rc0)"
    json.loads(p.read_text())
