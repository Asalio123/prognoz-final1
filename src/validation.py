"""Rolling-origin валидация по configs/validation.yaml (заморожено 04.10.2026).

Для origin t и горизонта h: train = месяцы [0, t), тест = месяц t+h-1 (точечный прогноз
на горизонт h). Модель всегда прогнозирует шаги t..t+h-1, оценивается только t+h-1 —
все горизонты из одного прогона (как в B3/R06: 12 прогонов вместо 30).
"""
import pandas as pd

HORIZONS = [1, 3, 6, 12]
ORIGINS = list(range(12, 24))


def folds(horizons=HORIZONS, origins=ORIGINS):
    """Генератор (origin, horizon, test_idx). test_idx = origin + h - 1, <= 23."""
    for o in origins:
        for h in horizons:
            t = o + h - 1
            if t <= 23:
                yield o, h, t
