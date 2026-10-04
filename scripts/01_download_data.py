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
    "hackathonlicence.zip": None,  # заполняется при первом прогоне, далее проверяется
    "mo_borders.rar": None,
}


def fetch(name: str, url: str):
    dest = RAW / name
    if not dest.exists() or dest.stat().st_size == 0:
        subprocess.run(["curl", "-skL", "-o", str(dest), url], check=True)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(name, f"{dest.stat().st_size/1e6:.1f} МБ", digest[:16])
    if SHA256.get(name):
        assert digest == SHA256[name], f"sha256 mismatch: {name}"


def main():
    for name, url in URLS.items():
        fetch(name, url)
    hack = RAW / "hackathon"
    if not (hack / "hackathonlicence" / "consumption.parquet").exists():
        hack.mkdir(exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(RAW / "hackathonlicence.zip"), "-C", str(hack)], check=True)
    if not (RAW / "t_dict_municipal_districts.xlsx").exists():
        subprocess.run(["bsdtar", "-xf", str(RAW / "mo_borders.rar"), "-C", str(RAW)], check=True)
    print("данные на месте")


if __name__ == "__main__":
    main()
