"""Загрузка сырых данных с официальных URL + проверка sha256 (воспроизводимость с нуля).

sberbank.com требует curl -k (сертификаты НУЦ). Сырые данные НЕ коммитятся (п.9.2.2 Положения).
"""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

URLS = {
    "hackathonlicence.zip": "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip",
    "mo_borders.rar": "https://www.sberbank.com/common/files/t_dict_municipal.rar",
}
SHA256 = {  # зафиксировано 04.10.2026 при первой загрузке
    "hackathonlicence.zip": "a9f932ff4096a7df797d1547987937f34d3995ac445b4748177114488d12b010",
    "mo_borders.rar": "319ed22684b77716641bc15b61f7325adc44e9fc47e9be2973d88412d27d21f5",
}


def fetch(name: str, url: str):
    dest = RAW / name
    if not dest.exists() or dest.stat().st_size == 0:
        subprocess.run(["curl", "-skL", "-o", str(dest), url], check=True)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(name, f"{dest.stat().st_size/1e6:.1f} МБ", digest[:16])
    if SHA256.get(name):
        assert digest == SHA256[name], f"sha256 mismatch: {name} ({digest} != {SHA256[name]})"


def verify_manifest():
    """Проверка configs/MANIFEST.sha256 для уже существующих файлов."""
    manifest = ROOT / "configs" / "MANIFEST.sha256"
    if not manifest.exists():
        return
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, rel = line.split(maxsplit=1)
        target = ROOT / rel
        if target.exists():
            got = hashlib.sha256(target.read_bytes()).hexdigest()
            assert got == expected, f"MANIFEST mismatch: {rel} ({got} != {expected})"


def setup_external_data():
    """Обеспечение наличия вспомогательных внешних рядов (ЦБ РФ, календарь, СберИндекс API)."""
    ext = ROOT / "data" / "external"
    ext.mkdir(parents=True, exist_ok=True)
    # Если запуск внутри воркспейса с общей data/external — гарантируем доступность
    sibling_ext = ROOT.parents[1] / "data" / "external"
    if sibling_ext.exists() and sibling_ext != ext:
        import shutil
        for item in ["cbr_keyrate_2023_2024.csv", "cbr_cpi_yoy_2023_2024.csv",
                     "cbr_usdrub_2023_2024.csv", "isdayoff_2023.txt", "isdayoff_2024.txt"]:
            src = sibling_ext / item
            dst = ext / item
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        sber_src = sibling_ext / "sberindex_api"
        sber_dst = ext / "sberindex_api"
        if sber_src.exists() and not sber_dst.exists():
            shutil.copytree(sber_src, sber_dst, dirs_exist_ok=True)


def main():
    for name, url in URLS.items():
        fetch(name, url)
    hack = RAW / "hackathon"
    if not (hack / "hackathonlicence" / "consumption.parquet").exists():
        hack.mkdir(exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(RAW / "hackathonlicence.zip"), "-C", str(hack)], check=True)
    if not (RAW / "t_dict_municipal_districts.xlsx").exists():
        subprocess.run(["bsdtar", "-xf", str(RAW / "mo_borders.rar"), "-C", str(RAW)], check=True)
    setup_external_data()
    verify_manifest()
    print("данные на месте, контрольные суммы проверены")


if __name__ == "__main__":
    main()
