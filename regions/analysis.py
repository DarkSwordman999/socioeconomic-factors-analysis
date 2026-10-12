"""Статистические процедуры исследования: проверки распределений, корреляции,
парные и множественные регрессии, диагностика, устойчивость, сходимость."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from regions.clusters import FEATURES, SEED, _matrix
from regions.indicators import CONTROLS, TARGETS
from regions.models import SPECS, _sample, fit_spec

FACTORS = ["gdp_real", "invest_real", "income_real"]
LN = {"gdp_real": "ln_gdp", "invest_real": "ln_invest", "income_real": "ln_income"}
FORMS = ["linear", "log", "power", "exp", "quadratic"]


# --- распределения -----------------------------------------------------------

def normality(panel: pd.DataFrame, year: int, columns) -> pd.DataFrame:
    """Критерий Шапиро — Уилка для показателей одного года: в уровнях и в логарифмах."""
    data = panel[panel["year"] == year]
    rows = []
    for col in columns:
        values = data[col].dropna()
        w, p = stats.shapiro(values)
        w_log, p_log = stats.shapiro(np.log(values)) if (values > 0).all() else (np.nan, np.nan)
        rows.append({"indicator": col, "n": len(values), "skew": values.skew(), "w": w, "p": p,
                     "skew_log": np.log(values).skew() if (values > 0).all() else np.nan,
                     "w_log": w_log, "p_log": p_log})
    return pd.DataFrame(rows)


def extremes(panel: pd.DataFrame, year: int, column: str, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel[panel["year"] == year].dropna(subset=[column])[["region", column]]
    return data.nlargest(n, column), data.nsmallest(n, column)


# --- корреляции --------------------------------------------------------------

def corr_tests(panel: pd.DataFrame, year: int) -> pd.DataFrame:
    """Корреляции факторов с социальными показателями в одном году: Пирсон (с ln x) и Спирмен."""
    data = panel[panel["year"] == year]
    rows = []
    for target in TARGETS:
        for factor in FACTORS + CONTROLS:
            pair = data[[factor, target]].dropna()
            x = np.log(pair[factor]) if factor in FACTORS else pair[factor]
            r, p = stats.pearsonr(x, pair[target])
            rho, p_rho = stats.spearmanr(pair[factor], pair[target])
            n = len(pair)
            rows.append({"target": target, "factor": factor, "n": n, "pearson": r,
                         "t": r * np.sqrt((n - 2) / (1 - r ** 2)), "p": p, "spearman": rho, "p_spearman": p_rho})
    return pd.DataFrame(rows)


# --- парные регрессии --------------------------------------------------------

def _pair_fit(x: np.ndarray, y: np.ndarray, form: str):
    """Оценивает одну форму связи; возвращает (параметры, функция прогноза, модель)."""
    if form == "linear":
        model = sm.OLS(y, sm.add_constant(x)).fit()
        a, b = model.params
        return (a, b), lambda v: a + b * v, model
    if form == "log":
        model = sm.OLS(y, sm.add_constant(np.log(x))).fit()
        a, b = model.params
        return (a, b), lambda v: a + b * np.log(v), model
    if form == "power":
        model = sm.OLS(np.log(y), sm.add_constant(np.log(x))).fit()
        a, b = np.exp(model.params[0]), model.params[1]
        return (a, b), lambda v: a * v ** b, model
    if form == "exp":
        model = sm.OLS(np.log(y), sm.add_constant(x)).fit()
        a, b = np.exp(model.params[0]), model.params[1]
        return (a, b), lambda v: a * np.exp(b * v), model
    model = sm.OLS(y, np.column_stack([np.ones_like(x), x, x ** 2])).fit()
    a, b, c = model.params
    return (a, b, c), lambda v: a + b * v + c * v ** 2, model


def _elasticity(form: str, params, x_mean: float, y_mean: float) -> float:
    if form == "linear":
        return params[1] * x_mean / y_mean
    if form == "log":
        return params[1] / y_mean
    if form == "power":
        return params[1]
    if form == "exp":
        return params[1] * x_mean
    return (params[1] + 2 * params[2] * x_mean) * x_mean / y_mean


def pair_models(panel: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    """Парные регрессии «фактор -> показатель» в пяти формах на срезе одного года.

    x — фактор в тыс. руб. R² и ошибка аппроксимации считаются в исходных единицах y,
    чтобы формы были сопоставимы. Возвращает таблицу и функции прогноза для графиков.
    """
    data = panel[panel["year"] == year]
    rows, curves = [], {}
    for target in TARGETS:
        for factor in FACTORS:
            pair = data[[factor, target]].dropna()
            x, y = pair[factor].to_numpy() / 1000, pair[target].to_numpy()
            n = len(pair)
            for form in FORMS:
                params, predict, model = _pair_fit(x, y, form)
                k = len(params) - 1
                fitted = predict(x)
                sse, sst = ((y - fitted) ** 2).sum(), ((y - y.mean()) ** 2).sum()
                r2 = 1 - sse / sst
                rows.append({
                    "target": target, "factor": factor, "form": form, "n": n,
                    "a": params[0], "b": params[1], "c": params[2] if k == 2 else np.nan,
                    "r2": r2, "r2_adj": 1 - (1 - r2) * (n - 1) / (n - k - 1),
                    "f": model.fvalue, "p": model.f_pvalue, "t_b": model.tvalues[1],
                    "mape": np.mean(np.abs((y - fitted) / y)) * 100,
                    "elasticity": _elasticity(form, params, x.mean(), y.mean()),
                })
                curves[(target, factor, form)] = predict
    table = pd.DataFrame(rows)
    table["best"] = table["r2_adj"] == table.groupby(["target", "factor"])["r2_adj"].transform("max")
    return table, curves


# --- множественная регрессия на срезе ----------------------------------------

def cross_section(panel: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Классическая множественная регрессия на срезе года с проверкой предпосылок МНК."""
    data = panel[panel["year"] == year].copy()
    for col, name in LN.items():
        data[name] = np.log(data[col])
    terms = list(LN.values())
    coefs, quality, fits = [], [], {}
    for target in TARGETS:
        sample = data[[target] + terms].dropna()
        model = smf.ols(f"{target} ~ " + " + ".join(terms), data=sample).fit()
        robust = model.get_robustcov_results("HC3")
        fits[target] = model
        for i, term in enumerate(model.params.index):
            if term == "Intercept":
                continue
            coefs.append({
                "target": target, "term": term, "coef": model.params[term], "se": model.bse[term],
                "t": model.tvalues[term], "p": model.pvalues[term], "p_hc3": robust.pvalues[i],
                "beta_std": model.params[term] * sample[term].std() / sample[target].std(),
                "elasticity": model.params[term] / sample[target].mean(),
                "vif": variance_inflation_factor(model.model.exog, i),
            })
        _, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
        jb, jb_p, _, _ = jarque_bera(model.resid)
        reset = linear_reset(model, power=2, use_f=True)
        quality.append({
            "target": target, "n": int(model.nobs), "intercept": model.params["Intercept"],
            "r2": model.rsquared, "r2_adj": model.rsquared_adj, "f": model.fvalue, "f_p": model.f_pvalue,
            "se_reg": np.sqrt(model.scale), "bp_p": bp_p, "jb_p": jb_p, "reset_p": float(reset.pvalue),
            "dw": durbin_watson(model.resid),
            "mape": np.mean(np.abs(model.resid / sample[target])) * 100,
        })
    return pd.DataFrame(coefs), pd.DataFrame(quality), fits


def yearly_coefficients(panel: pd.DataFrame) -> pd.DataFrame:
    """Та же регрессия отдельно для каждого года: как меняется коэффициент при доходах."""
    rows = []
    for year, data in panel.groupby("year"):
        data = data.copy()
        for col, name in LN.items():
            data[name] = np.log(data[col])
        for target in TARGETS:
            sample = data[[target] + list(LN.values())].dropna()
            model = smf.ols(f"{target} ~ ln_gdp + ln_invest + ln_income", data=sample).fit(cov_type="HC3")
            low, high = model.conf_int().loc["ln_income"]
            rows.append({"year": year, "target": target, "coef": model.params["ln_income"],
                         "ci_low": low, "ci_high": high, "r2": model.rsquared, "n": int(model.nobs)})
    return pd.DataFrame(rows)


# --- панельные модели: тесты и устойчивость -----------------------------------

def panel_tests(panel: pd.DataFrame) -> pd.DataFrame:
    """F-тест на значимость фиксированных эффектов, гетероскедастичность и автокорреляция остатков."""
    rows = []
    for target in TARGETS:
        fe, data, _ = fit_spec(panel, target, "fe")
        pooled = smf.ols(f"{target} ~ ln_gdp + ln_invest + ln_income", data=data).fit()
        q = pooled.df_resid - fe.df_resid
        f = ((pooled.ssr - fe.ssr) / q) / (fe.ssr / fe.df_resid)
        _, bp_p, _, _ = het_breuschpagan(pooled.resid, pooled.model.exog)

        resid = data.assign(e=fe.resid.to_numpy()).sort_values(["region", "year"])
        step = resid.groupby("region")["year"].diff()
        diff = resid.groupby("region")["e"].diff()[step == 1]
        lagged = resid.groupby("region")["e"].shift(1)[step == 1]
        rows.append({
            "target": target, "f_fe": f, "df1": int(q), "df2": int(fe.df_resid),
            "f_fe_p": stats.f.sf(f, q, fe.df_resid), "bp_p": bp_p,
            "dw_panel": (diff ** 2).sum() / (resid["e"] ** 2).sum(),
            "rho": np.corrcoef(resid["e"][step == 1], lagged)[0, 1],
        })
    return pd.DataFrame(rows)


def partial_regression(panel: pd.DataFrame, target: str, term: str = "ln_income") -> pd.DataFrame:
    """Очищает показатель и фактор от остальных регрессоров и фиксированных эффектов (теорема Фриша — Во)."""
    spec = SPECS["fe_controls"]
    data = _sample(panel, target, spec)
    others = [t for t in ("ln_gdp", "ln_invest", "ln_income") if t != term] + spec["extra"]
    rhs = " + ".join(others)
    y_res = smf.ols(f"{target} ~ {rhs}", data=data).fit().resid
    x_res = smf.ols(f"{term} ~ {rhs}", data=data).fit().resid
    return pd.DataFrame({"region": data["region"], "year": data["year"], "x": x_res, "y": y_res})


def _income_row(model, label: str, target: str, n_regions: int) -> dict:
    low, high = model.conf_int().loc["ln_income"]
    return {"variant": label, "target": target, "coef": model.params["ln_income"], "se": model.bse["ln_income"],
            "p": model.pvalues["ln_income"], "ci_low": low, "ci_high": high,
            "n": int(model.nobs), "regions": n_regions}


def robustness(panel: pd.DataFrame, leaders: list[str]) -> pd.DataFrame:
    """Коэффициент при ln доходов в панельной модели при разных выборках и спецификациях."""
    caucasus = panel.loc[panel["district"] == "Северо-Кавказский", "region"].unique()
    lagged = panel.sort_values(["region", "year"]).copy()
    lagged[FACTORS] = lagged.groupby("region")[FACTORS].shift(1)
    variants = {
        "Базовая модель": panel,
        "2000–2011 гг.": panel[panel["year"] <= 2011],
        "2012–2023 гг.": panel[panel["year"] >= 2012],
        "Без 2020–2021 гг.": panel[~panel["year"].isin([2020, 2021])],
        "Без регионов-лидеров": panel[~panel["region"].isin(leaders)],
        "Без Северного Кавказа": panel[~panel["region"].isin(caucasus)],
        "Факторы с лагом в один год": lagged,
    }
    rows = []
    for target in TARGETS:
        for label, subset in variants.items():
            model, data, _ = fit_spec(subset, target, "fe_controls")
            rows.append(_income_row(model, label, target, data["region"].nunique()))

        # модель в первых разностях: постоянные различия регионов уходят при вычитании
        data = _sample(panel, target, SPECS["fe_controls"]).sort_values(["region", "year"])
        cols = [target, "ln_gdp", "ln_invest", "ln_income"] + CONTROLS
        delta = data.groupby("region")[cols].diff()
        delta[["region", "year"]] = data[["region", "year"]]
        delta = delta[data.groupby("region")["year"].diff() == 1].dropna()
        groups = delta["region"].astype("category").cat.codes
        model = smf.ols(f"{target} ~ ln_gdp + ln_invest + ln_income + " + " + ".join(CONTROLS) + " + C(year)",
                        data=delta).fit(cov_type="cluster", cov_kwds={"groups": groups})
        rows.append(_income_row(model, "Первые разности", target, delta["region"].nunique()))
    return pd.DataFrame(rows)


def poverty_elasticity(panel: pd.DataFrame) -> pd.DataFrame:
    """Эластичность бедности по доходам: панельная модель с ln бедности в левой части."""
    periods = {"2000–2023 гг.": panel, "2000–2011 гг.": panel[panel["year"] <= 2011],
               "2012–2023 гг.": panel[panel["year"] >= 2012]}
    rows = []
    for label, subset in periods.items():
        data = _sample(subset, "poverty", SPECS["fe_controls"])
        data["ln_poverty"] = np.log(data["poverty"])
        groups = data["region"].astype("category").cat.codes
        model = smf.ols("ln_poverty ~ ln_gdp + ln_invest + ln_income + " + " + ".join(CONTROLS)
                        + " + C(region) + C(year)", data=data).fit(cov_type="cluster", cov_kwds={"groups": groups})
        row = _income_row(model, label, "poverty", data["region"].nunique())
        row["mean_poverty"] = data["poverty"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# --- сходимость регионов ------------------------------------------------------

def sigma_convergence(panel: pd.DataFrame, columns=("income", "income_real", "gdp_real", "poverty", "life_exp")) -> pd.DataFrame:
    """Коэффициент вариации показателей между регионами по годам, %."""
    grouped = panel.groupby("year")[list(columns)]
    return (grouped.std() / grouped.mean() * 100).reset_index()


def beta_convergence(panel: pd.DataFrame, first: int, last: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Регрессия среднегодового темпа роста на исходный уровень: g = a + b·ln(y0)."""
    span = last - first
    wide = panel[panel["year"].isin([first, last])].pivot(index="region", columns="year")
    rows, points = [], []
    for col in ("income_real", "gdp_real"):
        pair = wide[col].dropna()
        x, growth = np.log(pair[first]), np.log(pair[last] / pair[first]) / span
        model = sm.OLS(growth, sm.add_constant(x)).fit(cov_type="HC3")
        b = model.params.iloc[1]
        speed = -np.log(1 + b * span) / span
        rows.append({"indicator": col, "n": len(pair), "a": model.params.iloc[0], "b": b,
                     "se": model.bse.iloc[1], "p": model.pvalues.iloc[1], "r2": model.rsquared,
                     "speed_pct": speed * 100, "half_life": np.log(2) / speed})
        points.append(pd.DataFrame({"indicator": col, "region": pair.index, "start": pair[first].to_numpy(),
                                    "growth_pct": growth.to_numpy() * 100}))
    return pd.DataFrame(rows), pd.concat(points, ignore_index=True)


# --- кластеризация: выбор k и главные компоненты --------------------------------

def cluster_diagnostics(panel: pd.DataFrame, year: int, assignments: pd.DataFrame, k_range=range(1, 9)):
    """Инерция по k, нагрузки главных компонент и критерий Краскела — Уоллиса по кластерам."""
    frame = panel[panel["year"] == year].dropna(subset=FEATURES).reset_index(drop=True)
    matrix = _matrix(frame)
    inertia = pd.DataFrame({
        "k": list(k_range),
        "inertia": [KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(matrix).inertia_ for k in k_range],
    })
    pca = PCA(n_components=3, random_state=SEED).fit(matrix)
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=["pc1", "pc2", "pc3"]).reset_index(names="feature")
    explained = pca.explained_variance_ratio_

    rows = []
    for col in FEATURES:
        groups = [group[col].to_numpy() for _, group in assignments.groupby("cluster")]
        h, p = stats.kruskal(*groups)
        rows.append({"feature": col, "h": h, "p": p})
    return inertia, loadings, explained, pd.DataFrame(rows)
