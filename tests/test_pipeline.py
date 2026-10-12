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


def test_report_tables(panel):
    from regions.report import describe, within_year_corr

    stats = describe(panel).set_index("indicator")
    assert stats.loc["life_exp", "n"] > 1900 and 60 < stats.loc["life_exp", "mean"] < 75
    corr = within_year_corr(panel)
    assert corr.loc["income_real", "poverty"] < -0.6
    assert np.allclose(np.diag(corr), 1)


def test_pair_models(panel):
    from regions import analysis

    pairs, curves = analysis.pair_models(panel, LAST_YEAR)
    assert len(pairs) == 45 and pairs.groupby(["target", "factor"])["best"].sum().eq(1).all()
    best = pairs[pairs["best"]].set_index(["target", "factor"]).loc[("poverty", "income_real")]
    assert best["form"] == "power" and best["elasticity"] < -1
    assert curves[("poverty", "income_real", "power")](50.0) > 0


def test_convergence(panel):
    from regions import analysis

    table, points = analysis.beta_convergence(panel, FIRST_YEAR, LAST_YEAR)
    income = table.set_index("indicator").loc["income_real"]
    assert income["b"] < 0 and income["p"] < 0.01 and income["half_life"] > 0
    sigma = analysis.sigma_convergence(panel).set_index("year")
    assert sigma.loc[2012, "income_real"] < sigma.loc[FIRST_YEAR, "income_real"]


def test_robustness_and_elasticity(panel):
    from regions import analysis

    elasticity = analysis.poverty_elasticity(panel)
    assert (elasticity["coef"] < 0).all() and (elasticity["p"] < 0.01).all()
    leaders = cluster_regions(panel, LAST_YEAR)[0].query("cluster == 1")["region"].tolist()
    poverty = analysis.robustness(panel, leaders).query("target == 'poverty'")
    assert len(poverty) == 8 and (poverty["ci_high"] < 0).all()


def test_rank_correlations_and_concordance(panel):
    import pandas as pd

    from regions import analysis

    tests = analysis.corr_tests(panel, LAST_YEAR).set_index(["target", "factor"])
    row = tests.loc[("poverty", "income_real")]
    assert row["ci_low"] < row["pearson"] < row["ci_high"] < 0
    assert row["kendall"] < 0 and abs(row["kendall"]) < abs(row["spearman"])

    same = pd.DataFrame({"a": range(10), "b": range(10), "c": range(10)})
    assert analysis.kendall_w(same)["w"] == pytest.approx(1)
    conc = analysis.concordance(panel, LAST_YEAR)
    assert conc.iloc[0]["w"] > conc.iloc[1]["w"] and conc["w"].between(0, 1).all()

    partial = analysis.partial_corr(panel, LAST_YEAR).set_index(["target", "factor"])
    assert abs(partial.loc[("poverty", "gdp_real"), "partial"]) < abs(partial.loc[("poverty", "gdp_real"), "pairwise"])
