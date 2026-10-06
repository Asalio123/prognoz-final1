PY := .venv/bin/python
N_CHUNKS := 6

.PHONY: data panel stats prophet prophet-bfix lgbm cpd news radar ensemble tables report reproduce clean-outputs

data:        ## скачать сырые данные с официальных URL (CC BY-SA 4.0, sha256)
	$(PY) scripts/01_download_data.py

panel: data  ## собрать чистую панель (полные 24-мес ряды, фильтр Общепит <300₽)
	$(PY) -m src.panel

stats:       ## стат-стек × полная сетка origins (h=1:12, h=3:10, h=6:7, h=12:1), 6 чанков
	@for i in 0 1 2 3 4 5; do \
		nice -n 10 $(PY) scripts/02_run_stats.py $$i $(N_CHUNKS) > outputs/logs/stats_grid_chunk$$i.log 2>&1 & \
	done; wait
	@echo "stats done; логи outputs/logs/stats_grid_chunk*.log"

prophet:     ## Prophet A (default) и B (yearly=4, multiplicative) на сетке origins
	$(PY) scripts/03_run_prophet.py --version A
	$(PY) scripts/03_run_prophet.py --version B

prophet-bfix: ## Prophet-B* = честный splice (origin 12 → A): yearly=4 требует ≥365 дней train
	$(PY) scripts/03b_prophet_b_fixed.py

lgbm:        ## глобальный LightGBM (direct-per-horizon, objective l1, log-таргет)
	$(PY) scripts/05_run_lgbm.py

seasadj:     ## DriftAdjSNaive / SeasProfileHA (кандидаты с сезонными профилями)
	$(PY) scripts/08_run_seasadj.py

ensemble:    ## median-энсамбль (гейт включения — configs/decision.json)
	$(PY) scripts/07_run_ensemble.py

cpd:         ## CPD-стенд: калибровка порогов отдельно, измерение отдельно, фронтир F1×FAR
	$(PY) scripts/06_run_cpd.py
	$(PY) scripts/06b_cpd_frontier.py

downstream:  ## решающая метрика детекторов: ΔMAE при refit-от-брейка (каузальные брейки)
	$(PY) scripts/13_downstream_dmae.py

news:        ## новостные индексы + абляция новостей в LGBM
	$(PY) scripts/11_run_news.py
	$(PY) scripts/16_news_ablation_stats.py

news-cpd:    ## стыковка новости × CPD (burst × PELT на национальном ряду «Всего»)
	$(PY) scripts/15_run_news_cpd.py

chronos:     ## Chronos-2 zero-shot (Apache-2.0), MPS
	$(PY) scripts/09_run_chronos.py

timesfm:     ## TimesFM 2.5 zero-shot (Apache-2.0), CPU ~2.5 ч
	$(PY) scripts/10_run_timesfm.py

radar:       ## панельный радар (результат rc0: честный ноль — см. отчёт §4)
	$(PY) scripts/12_run_radar.py

tables:      ## T1/T2/T3: matched cells, разрезы, значимость (DM/Wilcoxon/bootstrap/Holm)
	$(PY) scripts/04_build_tables.py

test:        ## минимальные тесты целостности (утечки, схемы, matched cells)
	$(PY) -m pytest tests/ -q

reproduce: data panel stats prophet prophet-bfix seasadj lgbm ensemble chronos cpd downstream news news-cpd radar tables  ## полный прогон от сырых данных до таблиц (CPU, ~6–8 ч wall на 8 ядрах)
	@echo "reproduce done"

clean-outputs:  ## удалить outputs/preds_* (перед полной пересборкой)
	rm -f outputs/preds_*.parquet
