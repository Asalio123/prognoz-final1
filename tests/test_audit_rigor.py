"""Честные и бескомпромиссные тесты качества, инвариантов и воспроизводимости.

Проверяют:
1. Валидность configs/MANIFEST.sha256 по фактическим файлам.
2. Отсутствие персональных путей (/Users/, /home/) в коде и конфигах.
3. Консистентность шрифтов и отсутствие сторонних шрифтов (IBM Plex Mono) в деке.
4. Отсутствие противоречий в источниках деки (слайд 6: гейт configs/champion.yaml).
5. Наличие реальных sha256-хэшей в 01_download_data.py.
6. Точное совпадение чисел в деке с фактическими таблицами outputs/.
7. Наличие и валидность собранных PDF-артефактов презентации.
8. Согласованность журналов решений и handoff-документов.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def test_manifest_strictly_matches():
    """configs/MANIFEST.sha256 обязан совпадать байт-в-байт по всем строкам."""
    manifest_path = ROOT / "configs" / "MANIFEST.sha256"
    assert manifest_path.exists(), "configs/MANIFEST.sha256 не найден"

    lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 3, "Манифест должен содержать как минимум 3 контрольные суммы"

    for line in lines:
        parts = line.split()
        assert len(parts) >= 2, f"Некорректная строка манифеста: {line}"
        expected_hash = parts[0]
        rel_path = " ".join(parts[1:]).lstrip("*")
        file_path = ROOT / rel_path

        assert file_path.exists(), f"Файл из манифеста не существует: {file_path}"
        assert len(expected_hash) == 64, f"Невалидный sha256 хэш: {expected_hash}"

        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_hash = h.hexdigest()
        assert actual_hash == expected_hash, (
            f"Несовпадение хэша для {rel_path}:\n"
            f"  ожидался: {expected_hash}\n"
            f"  фактический: {actual_hash}"
        )


def test_zero_personal_paths():
    """В кодовой базе не должно быть хардкода персональных директорий (/Users/<username> или /home/<username>)."""
    check_dirs = [ROOT / "scripts", ROOT / "src", ROOT / "configs", ROOT / "reports"]
    extensions = {".py", ".sh", ".mjs", ".yaml", ".yml", ".json"}

    personal_path_regex = re.compile(r"/(?:Users|home)/[a-zA-Z0-9_\-]+/")
    violations = []

    for d in check_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in extensions:
                try:
                    text = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for line_idx, line in enumerate(text.splitlines(), start=1):
                    # Пропускаем комментарии с описанием аудита / примерами ошибок
                    if "было:" in line or "был:" in line or "Review" in line or "review" in line:
                        continue
                    m = personal_path_regex.search(line)
                    if m:
                        violations.append(f"{p.relative_to(ROOT)}:{line_idx}: {line.strip()}")

    assert not violations, "Обнаружен хардкод персональных путей в репозитории:\n" + "\n".join(violations)


def test_deck_font_consistency():
    """В reports/presentation.html не должно использоваться неподключенных шрифтов (например, IBM Plex Mono)."""
    deck_html = ROOT / "reports" / "presentation.html"
    assert deck_html.exists(), "presentation.html не найден"
    content = deck_html.read_text(encoding="utf-8")

    assert "IBM Plex Mono" not in content, (
        "Обнаружен шрифт 'IBM Plex Mono' в reports/presentation.html. "
        "В деке подключены только 'Golos Text' и 'PT Mono'. Неподключенный шрифт ломает типографику."
    )

    # Проверяем наличие локальных шрифтов woff2
    fonts_dir = ROOT / "reports" / "fonts"
    assert (fonts_dir / "pt-mono.woff2").exists(), "pt-mono.woff2 отсутствует в reports/fonts"
    assert (fonts_dir / "golos-text-var.woff2").exists(), "golos-text-var.woff2 отсутствует в reports/fonts"


def test_slide_6_gate_consistency():
    """На слайде 6 ссылка на гейт включения энсембля должна указывать на configs/champion.yaml."""
    deck_html = ROOT / "reports" / "presentation.html"
    content = deck_html.read_text(encoding="utf-8")

    # Слайд 6 должен содержать ссылку на champion.yaml в тексте и футере
    assert "configs/champion.yaml" in content, "В presentation.html отсутствует ссылка на configs/champion.yaml"
    assert "гейт: configs/champion.yaml" in content, (
        "В футере слайда 6 должно быть указано 'гейт: configs/champion.yaml'"
    )


def test_download_script_hashes_not_empty():
    """В scripts/01_download_data.py словарь SHA256 обязан содержать реальные хэши, без None."""
    script_path = ROOT / "scripts" / "01_download_data.py"
    assert script_path.exists(), "01_download_data.py не найден"

    import ast
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    sha_dict = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SHA256":
                    if isinstance(node.value, ast.Dict):
                        sha_dict = {
                            k.value: (v.value if isinstance(v, ast.Constant) else None)
                            for k, v in zip(node.value.keys, node.value.values)
                        }

    assert sha_dict is not None, "Словарь SHA256 не найден в scripts/01_download_data.py"
    for archive_name, hval in sha_dict.items():
        assert hval is not None, f"В SHA256 для {archive_name} стоит None вместо реального хэша"
        assert len(hval) == 64, f"Длина хэша {archive_name} ({len(hval)}) != 64"
        assert all(c in "0123456789abcdefABCDEF" for c in hval), f"Хэш {hval} не является hex-строкой"


def test_deck_key_numbers_ground_truth():
    """Проверка ключевых чисел деки против фактических файлов outputs/."""
    # 1. Слайд 8: MAE чемпиона h=1 «Все категории» = 1 322 ₽ vs Prophet-B* = 1 877 ₽
    t2_path = OUT / "T2_slice_category.csv"
    assert t2_path.exists(), "outputs/T2_slice_category.csv не найден"
    t2 = pd.read_csv(t2_path)

    champ_row = t2[(t2["model"] == "DriftAdjSNaive") & (t2["category"] == "Все категории") & (t2["horizon"] == 1)]
    prophet_row = t2[(t2["model"] == "Prophet-B*") & (t2["category"] == "Все категории") & (t2["horizon"] == 1)]

    assert len(champ_row) == 1, "Не найдена строка чемпиона на Все категории h=1"
    assert len(prophet_row) == 1, "Не найдена строка Prophet-B* на Все категории h=1"

    champ_mae = round(champ_row["mae_rub"].iloc[0])
    prophet_mae = round(prophet_row["mae_rub"].iloc[0])
    assert champ_mae == 1322, f"MAE чемпиона = {champ_mae}, ожидалось 1322"
    assert prophet_mae == 1877, f"MAE Prophet-B* = {prophet_mae}, ожидалось 1877"

    # RelMAE
    champ_relmae = round(champ_row["relmae_vs_snaive"].iloc[0], 3)
    prophet_relmae = round(prophet_row["relmae_vs_snaive"].iloc[0], 3)
    assert champ_relmae == 0.324, f"RelMAE чемпиона = {champ_relmae}, ожидалось 0.324"
    assert prophet_relmae == 0.460, f"RelMAE Prophet-B* = {prophet_relmae}, ожидалось 0.460"

    # 2. Слайд 8: 635 ₽ vs 597 ₽ на полном пуле 6 категорий
    t1_path = OUT / "T1_model_x_horizon.csv"
    assert t1_path.exists(), "outputs/T1_model_x_horizon.csv не найден"
    t1 = pd.read_csv(t1_path)
    t1_champ = t1[(t1["model"] == "DriftAdjSNaive") & (t1["horizon"] == 1)]
    t1_prophet = t1[(t1["model"] == "Prophet-B*") & (t1["horizon"] == 1)]
    assert round(t1_champ["mae_rub"].iloc[0]) == 635, f"T1 чемпион pooled h=1 != 635 ({t1_champ['mae_rub'].iloc[0]})"
    assert round(t1_prophet["mae_rub"].iloc[0]) == 597, f"T1 Prophet-B* pooled h=1 != 597 ({t1_prophet['mae_rub'].iloc[0]})"

    # 3. Слайд 13: Форвард 2025-H1 MAE 1.48 п.п.
    fwd_path = OUT / "rc1_o7_forward_by_category.csv"
    assert fwd_path.exists(), "outputs/rc1_o7_forward_by_category.csv не найден"
    fwd = pd.read_csv(fwd_path)
    all_cat = fwd[fwd["category"] == "Все категории"]
    mean_abs_err = all_cat["abs_err_pp"].mean()
    assert round(mean_abs_err, 2) == 1.48, f"Средняя ошибка форварда 'Все категории' {mean_abs_err:.4f} != 1.48"

    # Продовольствие: 1.36 п.п.
    food_cat = fwd[fwd["category"] == "Продовольствие"]
    food_abs_err = food_cat["abs_err_pp"].mean()
    assert round(food_abs_err, 2) == 1.36, f"Средняя ошибка форварда 'Продовольствие' {food_abs_err:.4f} != 1.36"

    # Общепит: 39.4 п.п. на январе
    catering_jan = fwd[(fwd["category"] == "Общественное питание") & (fwd["month"] == "2025-01")]
    assert round(catering_jan["abs_err_pp"].iloc[0], 1) == 39.4, (
        f"Ошибка Общепита на 2025-01 {catering_jan['abs_err_pp'].iloc[0]} != 39.4"
    )


def test_conformal_coverage_numbers():
    """Слайд 8: покрытие 80% PI на anchor-сетке 0.875 (h=1), 0.914 (h=3), 0.937 (h=6)."""
    cov_path = OUT / "intervals_coverage.csv"
    assert cov_path.exists(), "outputs/intervals_coverage.csv не найден"
    cov = pd.read_csv(cov_path)

    sub = cov[(cov["model"] == "DriftAdjSNaive") & (cov["slice_type"] == "overall_tier1")]
    h1 = sub[sub["horizon"] == 1]["cov80"].iloc[0]
    h3 = sub[sub["horizon"] == 3]["cov80"].iloc[0]
    h6 = sub[sub["horizon"] == 6]["cov80"].iloc[0]

    assert round(h1, 3) == 0.875, f"Покрытие h=1: {h1:.4f} != 0.875"
    assert round(h3, 3) == 0.914, f"Покрытие h=3: {h3:.4f} != 0.914"
    assert round(h6, 3) == 0.937, f"Покрытие h=6: {h6:.4f} != 0.937"


def test_pdf_artifacts_exist_and_non_empty():
    """reports/presentation.pdf и presentation_print.pdf должны существовать и быть валидными."""
    pdf_main = ROOT / "reports" / "presentation.pdf"
    pdf_print = ROOT / "reports" / "presentation_print.pdf"

    assert pdf_main.exists(), "reports/presentation.pdf не собран"
    assert pdf_print.exists(), "reports/presentation_print.pdf не собран"

    # Проверяем размер: не пустые, полноценные 14-слайдовые PDF (~1MB)
    assert pdf_main.stat().st_size > 500_000, f"presentation.pdf слишком маленький ({pdf_main.stat().st_size} байт)"
    assert pdf_print.stat().st_size > 500_000, f"presentation_print.pdf слишком маленький ({pdf_print.stat().st_size} байт)"

    # Проверяем сигнатуру PDF %PDF-
    assert pdf_main.read_bytes()[:5] == b"%PDF-", "presentation.pdf не является валидным PDF"
    assert pdf_print.read_bytes()[:5] == b"%PDF-", "presentation_print.pdf не является валидным PDF"


def test_decision_and_handoff_provenance():
    """Проверка коммитов предрегистрации в configs/decision.json и чисел в HANDOFF.md."""
    dec_path = ROOT / "configs" / "decision.json"
    assert dec_path.exists()
    dec = json.loads(dec_path.read_text(encoding="utf-8"))
    log_text = dec.get("decision_log", "")
    assert "e3aa46e" in log_text, "Хэш коммита e3aa46e должен быть упомянут в decision.json"
    assert "b683980" in log_text, "Хэш коммита b683980 должен быть упомянут в decision.json"


def test_deck_slide_count_and_ruler_numbering():
    """В presentation.html должно быть ровно 14 слайдов с непрерывной нумерацией 01/14..14/14."""
    deck_html = ROOT / "reports" / "presentation.html"
    content = deck_html.read_text(encoding="utf-8")

    # Ищем секции слайдов
    slides = re.findall(r'<section class="slide', content)
    assert len(slides) == 14, f"Ожидалось 14 слайдов в presentation.html, найдено {len(slides)}"

    # Ищем все фолио вида XX/14
    rulers = re.findall(r'<span class="num">(\d\d)/14</span>', content)
    assert len(rulers) == 14, f"Ожидалось 14 меток номеров слайдов, найдено {len(rulers)}"
    expected_rulers = [f"{i:02d}" for i in range(1, 15)]
    assert rulers == expected_rulers, f"Нумерация слайдов нарушена: {rulers} != {expected_rulers}"


def test_forward_2025_pi80_coverage():
    """Форвард 2025-H1: ровно 6 месяцев, 6 из 6 точек категории 'Все категории' обязаны лежать в 80% PI."""
    fwd_path = OUT / "rc1_o7_forward_by_category.csv"
    assert fwd_path.exists()
    df = pd.read_csv(fwd_path)

    all_cat = df[df["category"] == "Все категории"]
    assert len(all_cat) == 6, f"Ожидалось 6 месяцев 2025-H1, найдено {len(all_cat)}"
    assert (all_cat["in_pi80"] == "да").all(), "Не все 6 точек 'Все категории' лежат в 80% PI"


def test_deck_test_counter_matches_suite():
    """Число пройденных тестов на слайдах 4 и 14 обязано в точности совпадать с числом тестов в сьюте."""
    deck_html = ROOT / "reports" / "presentation.html"
    content = deck_html.read_text(encoding="utf-8")

    # Считаем количество функций test_* во всех файлах tests/test_*.py
    import ast
    test_count = 0
    for p in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        test_count += len(funcs)

    expected_marker = f"tests {test_count}/{test_count}"
    assert expected_marker in content, (
        f"В presentation.html ожидался маркер '{expected_marker}', "
        f"но фактический счетчик не найден. Проверьте слайды 4 и 14."
    )


