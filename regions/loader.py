"""Чтение таблиц сборника Росстата: «регион × год» из листа XLSX в длинный формат."""

import re
from pathlib import Path

import pandas as pd

from regions.indicators import Indicator

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

RUSSIA = "Российская Федерация"

# Автономные округа входят в состав Архангельской и Тюменской областей;
# в панели остаются области целиком, чтобы регионы не пересекались.
PARTS = {
    "Ненецкий автономный округ",
    "Ханты-Мансийский автономный округ – Югра",
    "Ямало-Ненецкий автономный округ",
    "Архангельская область без автономного округа",
    "Тюменская область без автономных округов",
}

# В исходных файлах встречаются латинские буквы внутри русских названий
_LATIN = str.maketrans("aceopxyABCEHKMOPTX", "асеорхуАВСЕНКМОРТХ")


def clean_name(value) -> str:
    """Приводит название территории к единому виду."""
    name = str(value).translate(_LATIN)
    name = re.sub(r"(\d\)[,;]?\s*)+$", "", name.strip())   # сноски «1)», «1);2)»
    name = re.sub(r"\s+[-–—]\s*|\s*[-–—]\s+", " – ", name)  # тире с пробелами
    return re.sub(r"\s+", " ", name).strip()


def kind_of(name: str) -> str:
    """Тип строки: country, district, part или region."""
    if name == RUSSIA or name.startswith("Валовой региональный продукт"):
        return "country"
    if "федеральный округ" in name:
        return "district"
    if name in PARTS:
        return "part"
    return "region"


def _year(cell):
    if isinstance(cell, (int, float)) and not pd.isna(cell) and 1990 <= cell <= 2100 and cell == int(cell):
        return int(cell)
    if isinstance(cell, str):
        match = re.match(r"\s*(\d{4})", cell)   # «2000/2001», «20231)»
        if match:
            return int(match.group(1))
    return None


def _header_row(table: pd.DataFrame) -> int:
    for i, row in enumerate(table.itertuples(index=False)):
        if sum(_year(cell) is not None for cell in row[1:]) >= 10:
            return i
    raise ValueError("не найдена строка с годами")


def read_indicator(ind: Indicator, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Возвращает таблицу показателя: name, kind, district, year, value."""
    files = sorted(Path(raw_dir).glob(f"Раздел {ind.section} -*.xlsx"))
    if not files:
        raise FileNotFoundError(f"нет файла раздела {ind.section} в {raw_dir}")
    table = pd.read_excel(files[0], sheet_name=ind.sheet, header=None)

    head = _header_row(table)
    years = {col: _year(cell) for col, cell in table.iloc[head].items() if col > 0 and _year(cell)}

    records, district, seen = [], None, set()
    for _, row in table.iloc[head + 1:].iterrows():
        raw = row[0]
        if not isinstance(raw, str) or not raw.strip():
            continue
        if raw.strip()[0] in "_¾—":      # дальше идут сноски
            break
        name = clean_name(raw)
        if name.startswith("в том числе"):
            continue
        kind = kind_of(name)
        if kind == "country":
            name = RUSSIA
        if kind == "district":
            district = name
        if name in seen:
            continue
        seen.add(name)
        values = pd.to_numeric(row[list(years)], errors="coerce")
        for col, value in values.items():
            if pd.notna(value):
                records.append((name, kind, district if kind in ("region", "part") else None, years[col], float(value)))
    return pd.DataFrame(records, columns=["name", "kind", "district", "year", "value"])
