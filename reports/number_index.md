# Number Index — трассируемость «число → источник» (R09 §2.2, приложение В1)

Аудит 06.10.2026. Каждое число methodology.md отображено в файл/команду, которая его воспроизводит.
Статусы: **OK** (совпадает с вычислением) / **ROUND** (совпадает с округлением) / **PARTIAL** (частично совпадает, см. примечание) / **PHANTOM** (не воспроизводится — нужна замена) / **EXTERNAL** (внешний источник, не из outputs).

| # | Число в тексте | Где (строка methodology.md) | Источник (файл/команда) | Статус | Проверка |
|---|---|---|---|---|---|
| 1 | 2 016 МО | §0 (данные) | `panel.territory_id.nunique()` на `data/panel/panel.parquet` | OK | 2016 |
| 2 | 24 месяца (2023-01–2024-12) | §0 | `panel.month_idx.nunique()` | OK | 24, min=0, max=23 |
| 3 | 6 категорий | §0 | `panel.category.nunique()` | OK | 6 |
| 4 | Из 2 190 МО исключены 174 | §0 | `pd.read_parquet('data/raw/hackathon/hackathonlicence/consumption.parquet').territory_id.nunique()` | OK | raw=2190, panel=2016, разница=174 |
| 5 | медиана < 300 ₽/мес → 414 МО исключены | §0 | `(2016 − T2_slice_category[T2.category=='Общественное питание'].n_series.iloc[0])` | OK | 2016−1602=414. NB: configs/decision.json:16 и validation.yaml комментарий говорят «445» — конфиги устарели, в тексте 414 правильно |
| 6 | 27 806 заголовков | §0, §6 | `wc -l data/external/news/headlines_*.jsonl` | OK | 14620+13186=27806 |
| 7 | +15.1–15.6% номинальный рост | §0 | вычисление из panel (см. строку 9) | **PARTIAL** | median=15.12%, pooled=15.12%, mean=14.80%. Верхняя граница 15.6 не воспроизводится ни одной агрегацией (max YoY=19.0 по месяцам, но это не «рост панели»). PHANTOM для «15.6» |
| 8 | ИПЦ 9.5% | §0 | внешний факт ЦБ РФ (2024) | EXTERNAL | — |
| 9 | +5–6% реальный рост | §0 | расчёт 15.1−9.5 | OK | 5.6 |
| 10 | 4.6–5.9% реальный р/г Sberindex weekly | §0 | внешний ряд `data/external/sberindex_api/consumer-spending-growth_parquet.parquet` | EXTERNAL | не пересчитывал |
| 11 | log1p(расходы), CV постоянен | §0 | configs/validation.yaml (`target: log1p_value`) | OK | — |
| 12 | origins 12..23, min_train=12, h∈{1,3,6,12} | §1 | configs/validation.yaml | OK | — |
| 13 | h=1→12 фолдов, h=3→10, h=6→7, h=12→1 | §1 | T1_model_x_horizon.csv `n_origins` | OK | 12/10/7/1 |
| 14 | 2 016 парных ошибок на h=12 | §1 | T1 h=12: n_series=11682 (pooled), 2016 (Все категории) | OK | T2 category Все категории n_series=2016 |
| 15 | 20 конфигураций в 5 слоях | §2 | T1 count distinct model | OK | 20 (21 с news_v2) |
| 16 | Prophet порог 730 дней, forecaster.py:1187 | §2 | `.venv/lib/python3.12/site-packages/prophet/forecaster.py:1187` | OK | строка `yearly_disable = last - first < pd.Timedelta(days=730)` |
| 17 | T1 pooled h=1: B*=0.457, Ensemble=0.458 | §2 таблица | outputs/T1_model_x_horizon.csv | ROUND | 0.4566, 0.4577 |
| 18 | T1 pooled h=3: B*=0.412, DriftAdj=0.515 | §2 | T1 | ROUND | 0.4115, 0.5151 |
| 19 | T1 pooled h=6: B*=0.521, DriftAdj=0.574 | §2 | T1 | ROUND | 0.5206, 0.5738 |
| 20 | T1 pooled h=12: A/B*=0.626, SeasProfileHA=0.871 | §2 | T1 | ROUND | 0.6257 (A=B*), 0.8707 |
| 21 | T1 «Wilcoxon p<1e-8» | §2 | T3_significance.csv (pooled w_p) | OK | h=1 DriftAdj vs B*: w_p=9.5e-51; все «лучшие наши» в T3 p<1e-8 |
| 22 | Все категории h=1: DriftAdj=0.324, B*=0.460 | §2 | T2_slice_category.csv | ROUND | 0.3242, 0.4601 |
| 23 | Все категории h=3: 0.361, 0.398 | §2 | T2 | ROUND | 0.3609, 0.3984 |
| 24 | Все категории h=6: 0.422, 0.494 | §2 | T2 | ROUND | 0.4219, 0.4941 |
| 25 | Все категории h=12: 0.513, 0.617 | §2 | T2 | ROUND | 0.5126, 0.6167 |
| 26 | Медианное отношение 0.66/0.92/0.89/0.78 | §2 | вычисление из preds (per-series median of MAE_a/MAE_b на Все категории) | OK | 0.658 / 0.921 / 0.893 / 0.776 |
| 27 | Wilcoxon p (Holm): 1.7e-241 / 4.0e-16 / 2.7e-58 / 4.9e-14 | §2 | вычисление из preds (см. примечание) | **PARTIAL** | h=1: 1.7e-241 = Wilcoxon на per-series RelMAE (без Holm); h=3/6: Holm×2/×3 на per-series MAE Wilcoxon (4.0e-16, 2.7e-58 ✓); h=12: 4.88e-14 (Holm×1) ✓. Смешанная конвенция: h=1 — RelMAE без Holm, h=3/6/12 — MAE + Holm |
| 28 | F2.1: 0.457 vs 0.473, 0.412 vs 0.470, 0.521 vs 0.558 | §2 | T1 | ROUND | B* vs A на h=1/3/6: 0.4566/0.4728, 0.4115/0.4699, 0.5206/0.5582 |
| 29 | F2.4 ΔMAE ≤ −1% гейт энсембля | §2 | configs/champion.yaml (`included_if: ΔMAE <= -1%`) | OK | — |
| 30 | калибровка 11 304 / инъекции 378 | §3 | cpd_calibration_v2.csv n_series_cal, cpd_injection_scores_v2.csv n | OK | 11304+378=11682 |
| 31 | «400 стратифицированных» инъекций | §3 | cpd_injection_scores_v2.csv n=378 | **PHANTOM** | Фактически 378 (после стратификации и отброса невалидных). Текст должен говорить 378 |
| 32 | 65.9% панели (январь артефакт наивной схемы) | §3 | пересчёт: `deseasonalize_old` + PELT pen=0.05 на выборке 500 → 63.6%; на полной панели не прогонял | **PARTIAL** | Число зависит от pen и полного прогона; при pen=0.05 на 500-сэмпле 63.6%, близко к 65.9%. Точное 65.9% не воспроизводится (нет зафиксированного скрипта/лога) |
| 33 | наша схема → 0.5% | §3 | outputs/cpd_controls_v2.csv | **PHANTOM** | Ни одна ячейка не равна 0.005. PELT январская-яма-2024 = 0.0000, BOCPD = 0.0272, PELT ноябрь-распродажи = 0.0039. Если «наша схема» = PELT-рабочая-точка, правильно **0.0%**; если «максимум по методам» — 2.7% (BOCPD). 0.5% ниоткуда не следует |
| 34 | якутские декабри 0.0% (амплитуда ×1.44) | §3 | cpd_controls_v2.csv: PELT/BinSeg/Window/EWMA/BOCPD Якутия-декабрь = 0.000; CUSUM = 0.0037 | OK (для PELT) | Амплитуда 1.44 — см. строку 70 |
| 35 | FAR ≤ 2/1000 МО-мес | §3 | cpd_injection_scores_v2.csv BOCPD far_per_1000=1.865 | OK | < 2 |
| 36 | BOCPD recall 5.6% [95% CI 3.8–7.6] на 1.6σ | §3 | cpd_injection_scores_v2.csv BOCPD level delta=1.6: recall=0.0556 | PARTIAL | point 5.6% ✓; CI: Clopper-Pearson k=21/n=378 → [3.5%, 8.4%], Wilson → [3.7%, 8.3%]. Указанный [3.8, 7.6] не воспроизводится стандартными методами |
| 37 | «на средней зоне FAR ~20/1000 лидируют Window/CUSUM» | §3 | cpd_frontier_v2.csv | **PHANTOM** | При FAR 10–30 лидирует BOCPD (recall 47% @15.5/1000 на 3σ, 26% @15.5/1000 на 1.6σ). Window/CUSUM уступают BOCPD и в средней зоне. Формулировка неверна |
| 38 | Богородский Маркетплейсы +402% (2024-02) | §3, §7 | panel.territory_id=1267, category=Маркетплейсы | OK | 308 → 1545 ₽ (month 12→13), +401.6% |
| 39 | паводок апр-2024 −1.7…−4.5% м/м | §3 | не проверял детально (внешний кейс) | EXTERNAL | — |
| 40 | F3.3 ΔMAE +1.1% на h=3 (refit) | §3, §7 | outputs/downstream_dmae_v2.csv (legacy_pen3, h=3, fallback=False): dmae_pooled_pct=1.102 | OK | 1.102 ≈ +1.1% |
| 41 | 42 404 соседские пары ≤80 км | §4, §7 | outputs/spatial_propagation.csv pairs_le80km | OK | 42404 |
| 42 | BOCPD лифт 1.08, p=0.77 | §4 | spatial_propagation.csv | ROUND | 1.079, 0.767 |
| 43 | CUSUM лифт 1.12, p=0.52 | §4 | spatial_propagation.csv | ROUND | 1.122, 0.517 |
| 44 | circular-shift ×1000 | §4 | scripts/14_spatial_propagation.py docstring + rng loop | OK | — |
| 45 | Унинский→Богородский 60.8 км, lead 4 мес | §4 | connection.parquet: tid 1297↔1267 distance=60.8 (highway); lead = month9→13 = 4 по данным panel | OK | — |
| 46 | PR-AUC 0.049 vs база 0.048 (нормальные фолды) | §4, §7 | outputs/radar_metrics.csv (mean pr_auc / base по folds кроме 11) | ROUND | mean pr_auc = 0.0488, mean base = 0.0482 |
| 47 | топ-30: 6/300 vs ~14.5 ожидаемых | §4 | radar_metrics.csv prec30 | PARTIAL | sum(prec30 × 30) = 6.99 ≈ 7 попаданий (не 6); expected = 0.0482 × 300 = 14.5 ✓ |
| 48 | база 62.6% на фолде-границе | §4, decision.json | radar_metrics.csv fold 11 base | ROUND | 0.6264 |
| 49 | среднее 0.104 | §4 | radar_metrics.csv mean pr_auc по всем 11 фолдам | OK | (0.110+0.661+0.039+0.041+0.125+0.037+0.055+0.028+0.021+0.019+0.014)/11 = 0.1045 |
| 50 | Chronos-2 zs 0.515/0.668/0.976/1.941 | §5 | T1 chronos2_zs_log relmae | ROUND | 0.5147/0.6678/0.9761/1.9409 |
| 51 | TimesFM 2.5 zs 0.492/0.632/0.916/1.890 | §5 | T1 timesfm25_zs_log relmae | ROUND | 0.4917/0.6317/0.9158/1.8903 |
| 52 | «обе TSFM в пятёрке из 20» на h=1 | §5 | T1 h=1 ranking | **PHANTOM** | chronos2_cov_log=6-я, timesfm25_zs_log=11-я, chronos2_zs_log=12-я. Ни одна TSFM не в топ-5 |
| 53 | ковариаты Δ −0.043/−0.076/−0.078 (h=1/3/6) | §5 | T1: chronos2_zs_log − chronos2_cov_log | OK | 0.5147−0.4720=−0.0427; 0.6678−0.5918=−0.0760; 0.9761−0.8984=−0.0777 |
| 54 | ковариаты win rate 60–64% | §5 | вычисление из preds_chronos2_{zs,cov}_log.parquet | OK | 59.9/63.9/61.2% (h=1/3/6); h=12: 40.5% (ухудшение ✓) |
| 55 | квантили coverage 94% при номинале 80% на h≥2 | §5 | outputs/chronos_quantile_coverage.csv | OK | mean h=2..12 = 94.1% |
| 56 | квантили h=1 узки: 60% | §5 | chronos_quantile_coverage.csv h=1 | ROUND | 59.5% ≈ 60% |
| 57 | холодный старт 9 МО, RelMAE 1.03, win rate 0.46 | §5 | outputs/coldstart_chronos.csv | OK | pooled RelMAE=1.029, weighted win_rate=0.460 |
| 58 | GIFT-Eval monthly 0.758 vs 0.765 AutoARIMA | §5 | внешний источник | EXTERNAL | — |
| 59 | контекст TSFM 8192 | §5 | документация Chronos-2 | EXTERNAL | — |
| 60 | κ=0.926, n=50 | §6 | outputs/news_kappa_v2.md («macro-κ по активным темам: 0.926») | OK | — |
| 61 | −6.1% MAE h=1 | §6 | outputs/news_ablation_stats_v2.csv h=1 delta_pct | ROUND | −6.072 |
| 62 | −11% на Продовольствии h=1 | §6 | news_ablation_by_category.csv | ROUND | −10.96 |
| 63 | Wilcoxon p=0.62 (Holm≈1.0) h=1 | §6 | news_ablation_stats_v2.csv wilcoxon_origin_p, holm_p_origin | OK | 0.622, 1.0 |
| 64 | +1.6% h=3 | §6 | news_ablation_stats_v2.csv | ROUND | 1.636 |
| 65 | +13.4% h=6 | §6 | news_ablation_stats_v2.csv | ROUND | 13.406 |
| 66 | Общепит +34%, Транспорт +12% на h=1 | §6 | news_ablation_by_category.csv | ROUND | +34.37, +11.95 |
| 67 | observed=3, p=0.13 | §6 | outputs/news_cpd_perm_v2.txt | ROUND | observed=3, p=0.1335 |
| 68 | ожидаемых ~1.8 по нулевой | §6 | outputs/news_cpd_perm_v2.txt | **PHANTOM** | v2: null mean=**1.35** (не 1.8). 1.8 — из старого v1 (news_cpd_perm.txt, observed=0, p=1.0 — битый ряд). Текст смешивает v1-null с v2-observed |
| 69 | ×10 000 пермутаций | §6 | news_cpd_perm_v2.txt | OK | 10000 reps |
| 70 | Находка 2: сезонные ×1.17/×1.21/×1.31/×1.17/×1.10, амплитуда 1.09–1.44 | §7 | пересчёт: median per-series log-deviation × exp | **PARTIAL** | Фактические: Продовольствие декабрь ×1.185 (заявл. 1.17), Маркетплейсы ноябрь ×1.33 (заявл. 1.21), Маркетплейсы декабрь ×1.54 (заявл. 1.31), Общепит июль ×1.21 (заявл. 1.17), Здоровье март ×1.11 (заявл. 1.10). Амплитуда декабря «Все категории»: 1.08–1.46 (заявл. 1.09–1.44). Max = Усть-Майский Якутия ×1.46 ✓ |
| 71 | Находка 1: 0.21 → 0.47 доля маркетплейсов | §7 | пересчёт: pooled=0.215→0.464; median per-MO=0.215→0.475 | ROUND | median даёт 0.475≈0.47 |
| 72 | Находка 1: 2.5% → 13.4% доля Богородского | §7 | пересчёт: 1545/(сумма Все категории Богородского 2024-02) | — | см. отчёт (не пересчитывал эту пару, требует per-MO суммы категорий) |
| 73 | Находка 1: 219 → 1323 (+504%), удержан 18 мес | §7 | panel tid=788 Общепит: 219 (month 4) → 1323 (month 5) | PARTIAL | +504% ✓; «18 мес» — фактически 19 мес (с 2023-06 по 2024-12 включительно) |
| 74 | Находка 4: +15.1% панель, 9.5% ИПЦ, 4.6–5.9% СИ | §7 | = строки 7–10 | OK | 15.12 ✓ |
| 75 | Находка 5: ΔRelMAE −0.04/−0.08/−0.08 | §7 | = строка 53 | OK | −0.043/−0.076/−0.078 (rounds to −0.04/−0.08/−0.08) |
| 76 | Находка 5: coverage 94% при 80%, h=1 60% | §7 | = строки 55–56 | OK | — |
| 77 | Находка 5: 9 МО, паритет с Naive | §7 | = строка 57 | OK | — |
| 78 | 5 сидов → mean±std | §8 | outputs/preds_lgbm{,_s137,_s271,_s314,_s1618}.parquet | OK | 5 файл-наборов: base(42)+137+271+314+1618 |
| 79 | make reproduce / sha256 / decision.json | §8 | Makefile, configs/MANIFEST.sha256, configs/decision.json | OK | — |
| 80 | «730 дней» порог Prophet | §2, Приложение А | = строка 16 | OK | — |

## Покрытие по разделам (полнота)

- §0 (данные) — покрыто (строки 1–10).
- §1 (валидация) — покрыто (11–14). Заявление «конфиги закоммичены до первого прогона» — ссылка на git history, не число.
- §2 (К2) — покрыто (15–29).
- §3 (К3) — покрыто (30–40).
- §4 (раннее выявление) — покрыто (41–49).
- §5 (К4 foundation) — покрыто (50–59). Исключения TimesFM 3.0/Moirai-2/TimeGPT — текстовые (лицензии), не числа.
- §6 (К5 новости) — покрыто (60–69).
- §7 (интерпретация) — покрыто (70–77).
- §8 (воспроизводимость) — покрыто (78–79).
- Приложение А (негативный ростер) — без чисел, только текст.
- Ограничения — без чисел.

Разделы/заявления без явного числа-источника: отсутствуют (все твёрдые числа трассированы).

---

# rc1-дополнение (07.10.2026): новые числа дня

| Число | Где | Источник | Статус |
|---|---|---|---|
| Покрытие 80% PI 0.875/0.914/0.937 (h=1/3/6, anchor tier-1) | §2 интервалы, Находка 4 | `make`-скрипт `scripts/23_champion_intervals.py` (Mondrian-пулы категория×квинтиль2023) → `outputs/intervals_coverage.csv` (slice overall_tier1) | OK (перегенерено 07.10; V1 пересчитал из parquet — совпало) |
| Q1 малые МО 0.734 → 0.833 | §2, Находка 4 | `intervals_coverage.csv` slice scale_quintile=Q1, DriftAdj h=1 | OK (0.8329) |
| Медианная ширина −3.1% (1957→1896 ₽) | §2 | review_rc1/O5 (rc1_o5_coverage_by_variant.csv) | OK |
| Декабрь-2024 покрытие 0.831 | §2, Находка 4 | `23_champion_intervals.py` вывод cal-month tm=23 (0.8312) | OK |
| Флаги риска: декабрь 82.6% (9647/11682) vs 35.5% | §2, Находка 4 | stdout `scripts/23_champion_intervals.py` (блок «Рискованный месяц») | OK |
| ноябрь-2024 второй пик 70.6% | Находка 4 | stdout `23_champion_intervals.py` (per_month 21: 0.706) | OK |
| Chronos |Δ| 0.134/0.112 vs наши 0.045/0.013 (h=6/12, origin 12) | §2 | `chronos_quantile_coverage.csv` + `intervals_coverage.csv` (origin 12) | OK |
| Форвард Общепит: MAE 39.4 п.п., 2/6 в PI80 | §2 форвард | `outputs/rc1_o7_forward_by_category.csv` | OK |
| Форвард Prophet-B*: 2.83 (агрегат) / 4.25 (Продовольствие) п.п. | §2 форвард | `scripts/rc1_o7_prophet_forward.py` → review_rc1/O7 | OK |
| Энсембли rc1: гейт не пройден; 50/50 full-grid −22% → holdout +46.5% | §2 F2.5а | `outputs/rc1_o1_*.csv` | OK |
| MCS-90 h=1: 8 моделей из 26, B* внутри; дно p<0.001 | §2 F2.5а | `outputs/rc1_o1_mcs*.csv` (block bootstrap 73 региона, 1000 реплик) | OK |
| Walk-forward селектор: закрывает 5.6–7.5% разрыва к оракулу; Маркетплейсы 0.419 vs 0.281 | §2 F2.5б | `outputs/rc1_o2_*.csv` | OK |
| CPD голосование: 10/10 проигрыш BOCPD (F1 0.271 vs 0.303) | §3 F3.1б | `outputs/rc1_o4_ensemble_matched.csv` | OK |
| BOCPD∩Window = 0 из 534 брейков | §3 F3.1б | `outputs/rc1_o4_realpanel_votes.csv` | OK |
| MDE spatial: лифт ≥×2.46 (BOCPD) / ≥×1.61 (CUSUM) при мощности 80% | §4 F4.2, Находка 6 | `outputs/rc1_o4_mde.csv` | OK |
| Refit: доля побед 48.0% (n=98), MDE ±14 п.п. | §3 F3.3, Находка 6 | `outputs/rc1_o4_mde.csv` + `downstream_dmae_v2.csv` | OK |
| burst×CPD: observed 3, симметричный нуль mean 3.69, p=0.84 (100k reps); MDE ≥8 совпадений для p<0.05 | §6 | `outputs/news_cpd_perm_v2.txt` (перегенерирован 07.10 после фикса окна) + Poisson(3.69) | OK |
| Linlin: чемпионы стабильны k∈[1;10]; сдвиг τ=0.67 при k=2: −20…−34% pooled h=1–3 | §1 | `outputs/rc1_o8_linlin.csv`, `rc1_o8_shift.csv` | OK |
| Маркетплейсы +54%/год (медиана), ×3.6 к pooled-дрейфу | §2, Находка 2 | `outputs/rc1_s6_verification.csv` (годовые суммы 54.0/54.2%) | OK |
| Офиц. реальный ряд 4.6–6.7% (Jul–Dec 2024) | Находка 1 | `outputs/external_real_growth.csv` | OK |
| Wilcoxon h=1 p=1.4e-241 (raw 3.43e-242 × Holm-4) | §2 T1 | `outputs/T3_significance.csv` + Holm×4 (V1: напечатанное ранее 1.7e-241 было ×5 — исправлено) | OK |
| CI recall 5.6%: Wilson [3.7–8.3] | §3 F3.1 | пересчёт 21/378 (V1; ранее [3.8–7.6] не воспроизводилось — исправлено) | OK |
