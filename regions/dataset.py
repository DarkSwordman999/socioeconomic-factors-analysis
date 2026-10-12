"""Сборка панели «регион × год» и пересчёт денежных показателей в реальные."""

from pathlib import Path

import pandas as pd

from regions.indicators import INDICATORS, MONEY
from regions.loader import RAW_DIR, read_indicator

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PANEL_PATH = PROCESSED_DIR / "panel.csv"
RUSSIA_PATH = PROCESSED_DIR / "russia.csv"

FIRST_YEAR, LAST_YEAR = 2000, 2023
BASE_YEAR = 2023


def build_panel(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Собирает панель по регионам и ряд по России в целом.

    Денежные показатели делятся на индекс цен региона: стоимость фиксированного
    набора товаров и услуг в регионе в данном году, отнесённая к его стоимости
    в среднем по России в базовом году. Так убираются и инфляция, и разница
    в уровне цен между регионами.
    """
    tables = {ind.key: read_indicator(ind, raw_dir) for ind in INDICATORS}

    def wide(kind: str) -> pd.DataFrame:
        parts = [
            t[t["kind"] == kind].set_index(["name", "year"])["value"].rename(key)
            for key, t in tables.items()
        ]
        frame = pd.concat(parts, axis=1).reset_index()
        return frame[frame["year"].between(FIRST_YEAR, LAST_YEAR)].copy()

    regions, russia = wide("region"), wide("country")

    base_cost = russia.loc[russia["year"] == BASE_YEAR, "basket"].iloc[0]
    for frame in (regions, russia):
        frame["price_index"] = frame["basket"] / base_cost
        for key in MONEY:
            frame[f"{key}_real"] = frame[key] / frame["price_index"]

    income = tables["income"]
    districts = income[income["kind"] == "region"].drop_duplicates("name").set_index("name")["district"]
    regions.insert(1, "district", regions["name"].map(districts).str.replace(" федеральный округ", ""))

    regions = regions.rename(columns={"name": "region"}).sort_values(["region", "year"]).reset_index(drop=True)
    russia = russia.drop(columns="name").sort_values("year").reset_index(drop=True)
    return regions, russia


def save_panel(regions: pd.DataFrame, russia: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    regions.round(4).to_csv(PANEL_PATH, index=False)
    russia.round(4).to_csv(RUSSIA_PATH, index=False)


def load_panel() -> pd.DataFrame:
    return pd.read_csv(PANEL_PATH)


def load_russia() -> pd.DataFrame:
    return pd.read_csv(RUSSIA_PATH)
