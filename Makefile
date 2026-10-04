PY := .venv/bin/python

.PHONY: data panel stats prophet smoke test reproduce

data:        ## скачать сырые данные с официальных URL (CC BY-SA 4.0, sha256)
	$(PY) scripts/01_download_data.py

panel:       ## собрать чистую панель (только полные 24-мес ряды, фильтр <300 руб)
	$(PY) -m src.panel

stats:       ## статистический стек: 8 моделей × 12 origins × 4 горизонта
	$(PY) scripts/02_run_stats.py

prophet:     ## Prophet A/B (default + честный) на согласованной сетке
	$(PY) scripts/03_run_prophet.py

test:
	$(PY) -m pytest tests/ -q

reproduce: data panel stats
	@echo "ядро воспроизведено; дальше make prophet"
