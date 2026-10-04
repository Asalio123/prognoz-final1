"""Метрики по configs/validation.yaml. MAE — контрактная (в рублях, после expm1).
sMAPE/MAPE осознанно исключены (см. configs/validation.yaml#excluded)."""
import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def per_series_table(err: pd.DataFrame) -> pd.DataFrame:
    """err: колонки [series_id, horizon, y_true, y_pred, y_snaive] (уровни рублей).
    Возвращает по сериям: mae, mae_snaive, median_level (для nMAE)."""
    g = err.copy()
    g["ae"] = (g["y_true"] - g["y_pred"]).abs()
    g["ae_snaive"] = (g["y_true"] - g["y_snaive"]).abs()
    return g.groupby(["series_id", "horizon"]).agg(
        mae=("ae", "mean"), mae_snaive=("ae_snaive", "mean"),
        median_level=("y_true", "median")).reset_index()


def pooled_summary(ps: pd.DataFrame) -> pd.Series:
    """Сводка по пулу серий одного (model, horizon): контрактная MAE, RelMAE, nMAE."""
    w = ps["mae_snaive"].sum()
    return pd.Series({
        "mae_rub": ps["mae"].mean(),  # по сериям (не по точкам) — устойчиво к крупным МО
        "relmae_vs_snaive": ps["mae"].sum() / w if w else np.nan,
        "median_nmae": (ps["mae"] / ps["median_level"]).median(),
    })
