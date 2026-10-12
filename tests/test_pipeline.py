import numpy as np
import pytest

from regions.clusters import cluster_regions
from regions.dataset import BASE_YEAR, FIRST_YEAR, LAST_YEAR, build_panel
from regions.loader import clean_name, kind_of
from regions.models import run_models


@pytest.fixture(scope="module")
def data():
    return build_panel()


@pytest.fixture(scope="module")
def panel(data):
    return data[0]


def test_clean_name():
    assert clean_name("Южный федеральный округ3)") == "Южный федеральный округ"
    assert clean_name("Республика Северная Осетия -Алания") == "Республика Северная Осетия – Алания"
    assert clean_name("Калинингpадская  область ") == "Калининградская область"   # латинская «p»
    assert clean_name("Кабардино-Балкарская Республика") == "Кабардино-Балкарская Республика"


def test_kind_of():
    assert kind_of("Российская Федерация") == "country"
    assert kind_of("Уральский федеральный округ") == "district"
    assert kind_of("Ямало-Ненецкий автономный округ") == "part"
    assert kind_of("Чукотский автономный округ") == "region"


def test_panel_shape(panel):
    assert panel["region"].nunique() == 82
    assert panel["year"].min() == FIRST_YEAR and panel["year"].max() == LAST_YEAR
    assert not panel.duplicated(["region", "year"]).any()
    assert panel["district"].notna().all() and panel["district"].nunique() == 8


def test_panel_has_no_aggregates(panel):
    names = set(panel["region"])
    assert "Российская Федерация" not in names
    assert not any("федеральный округ" in name for name in names)
    assert "Тюменская область" in names and "Ямало-Ненецкий автономный округ" not in names


def test_values_match_source(panel):
    row = panel[(panel["region"] == "Белгородская область") & (panel["year"] == 2000)].iloc[0]
    assert row["gdp"] == pytest.approx(27969.5)
    assert row["income"] == pytest.approx(1555)
    assert row["poverty"] == pytest.approx(33.6)
    assert row["life_exp"] == pytest.approx(68.07)


def test_deflator(data):
    panel, russia = data
    base = russia[russia["year"] == BASE_YEAR].iloc[0]
    assert base["price_index"] == pytest.approx(1)
    assert base["income_real"] == pytest.approx(base["income"])
    # в 2000 году цены были ниже, значит реальные доходы выше номинальных
    early = panel[panel["year"] == 2000].dropna(subset=["income_real"])
    assert (early["income_real"] > early["income"]).all()


def test_models(panel):
    coefs, summary = run_models(panel)
    assert len(summary) == 12 and summary["n"].min() > 1800
    income = coefs[(coefs["target"] == "poverty") & (coefs["spec"] == "fe") & (coefs["term"] == "ln_income")].iloc[0]
    assert income["coef"] < 0 and income["pvalue"] < 0.01
    within = summary[summary["spec"] == "fe"]["r2_within"]
    assert within.between(0, 1).all()


def test_clusters(panel):
    assignments, profiles, scores = cluster_regions(panel, LAST_YEAR)
    assert len(assignments) == 82 and set(assignments["cluster"]) == {1, 2, 3}
    assert profiles["income_real"].is_monotonic_decreasing   # нумерация по убыванию дохода
    assert np.isfinite(scores["silhouette"]).all()
