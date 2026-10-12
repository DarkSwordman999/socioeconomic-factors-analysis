"""Регрессии: от «наивной» модели в текущих ценах к панельной с фиксированными эффектами."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from regions.indicators import CONTROLS, TARGETS

# Спецификации: какие факторы логарифмируются и что добавляется в правую часть
SPECS = {
    "naive": {
        "title": "Текущие цены, общая выборка",
        "logs": ["gdp", "invest", "income"],
        "extra": ["year"],
    },
    "real": {
        "title": "Реальные цены, общая выборка",
        "logs": ["gdp_real", "invest_real", "income_real"],
        "extra": ["year"],
    },
    "fe": {
        "title": "Реальные цены, эффекты региона и года",
        "logs": ["gdp_real", "invest_real", "income_real"],
        "extra": ["C(region)", "C(year)"],
    },
    "fe_controls": {
        "title": "То же с контрольными переменными",
        "logs": ["gdp_real", "invest_real", "income_real"],
        "extra": CONTROLS + ["C(region)", "C(year)"],
    },
}

FACTOR_OF = {"gdp": "gdp", "gdp_real": "gdp", "invest": "invest", "invest_real": "invest",
             "income": "income", "income_real": "income"}


def _sample(panel: pd.DataFrame, target: str, spec: dict) -> pd.DataFrame:
    plain = [term for term in spec["extra"] if not term.startswith("C(")]
    data = panel[["region", "year", target] + spec["logs"] + [c for c in plain if c != "year"]].dropna().copy()
    for col in spec["logs"]:
        data[f"ln_{FACTOR_OF[col]}"] = np.log(data[col])
    return data


def _fit(formula: str, data: pd.DataFrame):
    groups = data["region"].astype("category").cat.codes
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups})


def fit_spec(panel: pd.DataFrame, target: str, spec_name: str):
    """Оценивает одну модель; стандартные ошибки кластеризованы по регионам."""
    spec = SPECS[spec_name]
    data = _sample(panel, target, spec)
    terms = [f"ln_{FACTOR_OF[col]}" for col in spec["logs"]] + spec["extra"]
    model = _fit(f"{target} ~ " + " + ".join(terms), data)

    effects = [term for term in spec["extra"] if term.startswith("C(")]
    r2_within = np.nan
    if effects:   # доля разброса, объяснённая сверх фиксированных эффектов
        base = _fit(f"{target} ~ " + " + ".join(effects), data)
        r2_within = 1 - model.ssr / base.ssr
    return model, data, r2_within


def run_models(panel: pd.DataFrame, targets=TARGETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает таблицу коэффициентов и сводку качества по всем моделям."""
    coefs, summary = [], []
    for target in targets:
        for name, spec in SPECS.items():
            model, data, r2_within = fit_spec(panel, target, name)
            ci = model.conf_int()
            for term in model.params.index:
                if term == "Intercept" or term.startswith("C("):
                    continue
                coefs.append({
                    "target": target, "spec": name, "term": term,
                    "coef": model.params[term], "se": model.bse[term], "pvalue": model.pvalues[term],
                    "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
                })
            summary.append({
                "target": target, "spec": name, "title": spec["title"],
                "n": int(model.nobs), "regions": data["region"].nunique(),
                "r2": model.rsquared, "r2_within": r2_within,
            })
    return pd.DataFrame(coefs), pd.DataFrame(summary)


def effect_of_10pct(coefs: pd.DataFrame) -> pd.DataFrame:
    """Пересчитывает коэффициенты при логарифмах в эффект роста фактора на 10 %."""
    logs = coefs[coefs["term"].str.startswith("ln_")].copy()
    for col in ("coef", "ci_low", "ci_high"):
        logs[col] = logs[col] * np.log(1.1)
    return logs.rename(columns={"coef": "effect"}).drop(columns="se")


def vif_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Коэффициенты вздутия дисперсии для логарифмов реальных факторов."""
    data = np.log(panel[["gdp_real", "invest_real", "income_real"]].dropna())
    matrix = np.column_stack([np.ones(len(data)), data.to_numpy()])
    return pd.DataFrame({
        "factor": data.columns,
        "vif": [variance_inflation_factor(matrix, i + 1) for i in range(data.shape[1])],
    })
