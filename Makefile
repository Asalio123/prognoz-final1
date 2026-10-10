PY := .venv/bin/python
N_CHUNKS := 6

.PHONY: data panel news-corpus stats prophet prophet-bfix lgbm seasadj ensemble cpd downstream news news-cpd chronos timesfm radar tables figures test verify reproduce clean-outputs

data:        ## скачать сырые данные СберИндекса с официальных URL (CC BY-SA 4.0, sha256)
	$(PY) scripts/01_download_data.py

news-corpus: ## новостный корпус заголовков Ъ/Ведомости 2023–24 (~25–40 мин)
	mkdir -p data/external/news
	$(PY) scripts/crawl_news_headlines.py

panel: data  ## собрать чистую панель (полные 24-мес ряды, фильтр Общепит <300₽)
	$(PY) -m src.panel

stats:       ## стат-стек × полная сетка origins (h=1:12, h=3:10, h=6:7, h=12:1), 6 чанков
	mkdir -p outputs/logs
	@fail=0; for i in 0 1 2 3 4 5; do \
		nice -n 10 $(PY) scripts/02_run_stats.py $$i $(N_CHUNKS) > outputs/logs/stats_grid_chunk$$i.log 2>&1 || fail=1 & \
	done; wait; [ $$fail -eq 0 ] || { echo "STATS FAILED — см. outputs/logs/"; exit 1; }
	@echo "stats done; логи outputs/logs/stats_grid_chunk*.log"

prophet:     ## Prophet A (default) и B (yearly=4, multiplicative) на сетке origins
	$(PY) scripts/03_run_prophet.py --version A
	$(PY) scripts/03_run_prophet.py --version B

prophet-bfix: ## Prophet-B* = честный splice (origin 12 → A): yearly=4 требует ≥365 дней train
	$(PY) scripts/03b_prophet_b_fixed.py

lgbm:        ## глобальный LightGBM (direct-per-horizon, objective l1, log-таргет)
	$(PY) scripts/05_run_lgbm.py
	$(PY) scripts/05_run_lgbm.py --news 1 --tag _v2

seasadj:     ## DriftAdjSNaive / SeasProfileHA (чемпионы по сезонным профилям и дрейфу)
	$(PY) scripts/08_run_seasadj.py

ensemble:    ## median-энсамбль (гейт включения — configs/champion.yaml)
	$(PY) scripts/07_run_ensemble.py

cpd:         ## CPD-стенд: disjoint калибровка/измерение, фронтир, контролы
	$(PY) scripts/06_run_cpd.py
	$(PY) scripts/06b_cpd_frontier.py
	$(PY) scripts/06c_cpd_controls.py

downstream:  ## решающая метрика детекторов: ΔMAE при refit-от-брейка (каузальные брейки)
	$(PY) scripts/13_downstream_dmae.py

news: news-corpus  ## новостные индексы + абляция со статистикой
	$(PY) scripts/11_run_news.py
	$(PY) scripts/16_news_ablation_stats.py

news-cpd:    ## стыковка новости × CPD (burst × PELT на нац. ряде «Всего»)
	$(PY) scripts/15_run_news_cpd.py

chronos:     ## Chronos-2 zero-shot + ковариаты (Apache-2.0), MPS/CPU
	$(PY) scripts/09_run_chronos.py
	$(PY) scripts/18_run_chronos_cov.py

timesfm:     ## TimesFM 2.5 zero-shot (Apache-2.0), CPU ~3.3 ч
	$(PY) scripts/10_run_timesfm.py

radar:       ## панельный радар (результат rc0: честный ноль — см. отчёт §4)
	$(PY) scripts/12_run_radar.py

tables:      ## T1/T2/T3: matched cells, разрезы, значимость (DM/Wilcoxon/bootstrap/Holm)
	$(PY) scripts/04_build_tables.py

deck:        ## проверка наличия готовой презентации reports/presentation.pdf
	@test -f reports/presentation.pdf && echo "reports/presentation.pdf готов (14 слайдов)" || { echo "presentation.pdf отсутствует"; exit 1; }

figures:     ## 7 фигур отчёта в reports/figures/
	$(PY) scripts/17_figures.py

test:        ## тесты целостности (утечки, matched cells, схемы, новостный гейт)
	$(PY) -m pytest tests/ -q

verify:      ## мгновенная сверка заглавных чисел отчёта из outputs/headline/ (~0.5 сек)
	$(PY) scripts/verify_headline.py

# Полный прогон от сырых данных до таблиц и фигур. Реалистично: ~9–11 ч wall на 8 ядрах
# (доминируют stats ~2.6 ч и prophet ~4.5–5 ч; timesfm ~3 ч CPU — можно закомментировать).
reproduce: data panel news-corpus stats prophet prophet-bfix seasadj lgbm ensemble chronos timesfm cpd downstream news news-cpd radar tables test figures
	@echo "reproduce done"

# Аудит-расширения rc1 (не входят в reproduce): форвард-2025 — scripts/28_forecast_2025.py;
# Mondrian-интервалы — rc1_o5_mondrian.py; энсембли-holdout — rc1_o1_*.py; селектор — rc1_o2_*;
# TSFM-энсембль/квантили — rc1_o3_*; голосование/MDE — rc1_o4_*; пространство — 14;
# matched-rate/NAB/ClaSP/тайминги — 26*; лестница цены — 27; linlin — rc1_o8_*;
# форвард-таблицы — rc1_o7_*; внешняя валидация — 24. Запуск поверх готового outputs/.

smoke: data panel test  ## быстрый контур без прогонов (тесты умеют skip без outputs)
	@echo "smoke done"
