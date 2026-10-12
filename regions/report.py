"""Сборка отчёта results/REPORT.md: все шаги анализа с таблицами и графиками."""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.stattools import jarque_bera

from regions.clusters import FEATURES, N_CLUSTERS
from regions.dataset import BASE_YEAR, FIRST_YEAR, LAST_YEAR
from regions.indicators import LABELS, TARGETS, UNITS
from regions.models import SPECS, effect_of_10pct

REPORT_PATH = Path(__file__).resolve().parent.parent / "results" / "REPORT.md"

DESCRIBED = ["gdp_real", "invest_real", "income_real", "poverty", "life_exp", "housing",
             "infant_mort", "unemployment", "doctors", "urban", "elderly", "students"]
LOGGED = ["gdp_real", "invest_real", "income_real"]
TERMS = {
    "ln_gdp": "ln ВРП", "ln_invest": "ln инвестиций", "ln_income": "ln доходов", "year": "Год (тренд)",
    "unemployment": "Безработица, %", "doctors": "Врачи на 10 тыс.", "urban": "Горожане, %",
    "elderly": "Пожилые, %",
}


CLUSTER_COLUMNS = {
    "income_real": "Доходы, руб.", "gdp_real": "ВРП, руб.", "poverty": "Бедность, %", "life_exp": "Жизнь, лет",
    "housing": "Жильё, кв. м", "unemployment": "Безраб., %", "doctors": "Врачи", "urban": "Горожане, %",
    "elderly": "Пожилые, %",
}


def _num(value, digits=2) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:.{digits}f}"


def _table(frame: pd.DataFrame, text_columns: int = 1) -> str:
    """Таблица Markdown из DataFrame; первые text_columns столбцов выравниваются влево."""
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join(["---"] * text_columns + ["---:"] * (len(frame.columns) - text_columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def describe(panel: pd.DataFrame) -> pd.DataFrame:
    """Описательная статистика по всем наблюдениям панели."""
    rows = []
    for key in DESCRIBED:
        values = panel[key].dropna()
        rows.append({
            "indicator": key, "n": len(values), "mean": values.mean(), "median": values.median(),
            "std": values.std(), "min": values.min(), "max": values.max(),
            "cv_pct": values.std() / values.mean() * 100, "skew": values.skew(), "kurtosis": values.kurtosis(),
        })
    return pd.DataFrame(rows)


def within_year_corr(panel: pd.DataFrame) -> pd.DataFrame:
    """Корреляции между регионами: из каждого показателя вычтено среднее за год."""
    data = panel[["year"] + DESCRIBED].copy()
    data[LOGGED] = np.log(data[LOGGED])
    return (data[DESCRIBED] - data.groupby("year")[DESCRIBED].transform("mean")).corr()


def residual_tests(fits: dict) -> pd.DataFrame:
    rows = []
    for target, model in fits.items():
        jb, p_value, skew, kurt = jarque_bera(model.resid)
        rows.append({"target": target, "resid_std": model.resid.std(), "skew": skew,
                     "kurtosis": kurt - 3, "jarque_bera": jb, "pvalue": p_value})   # эксцесс, как в describe
    return pd.DataFrame(rows)


def _describe_section(panel: pd.DataFrame, stats: pd.DataFrame) -> str:
    shown = pd.DataFrame({
        "Показатель": [f"{LABELS[k]}, {UNITS[k]}" for k in stats["indicator"]],
        "N": stats["n"],
        "Среднее": stats["mean"].map(_num), "Медиана": stats["median"].map(_num),
        "Ст. откл.": stats["std"].map(_num), "Мин.": stats["min"].map(_num), "Макс.": stats["max"].map(_num),
        "Вариация, %": stats["cv_pct"].map(lambda v: _num(v, 0)),
        "Асимметрия": stats["skew"].map(_num), "Эксцесс": stats["kurtosis"].map(_num),
    })
    shown["N"] = shown["N"].astype(str)
    first, last = (panel[panel["year"] == y][DESCRIBED].median() for y in (FIRST_YEAR, LAST_YEAR))
    change = pd.DataFrame({
        "Показатель": [f"{LABELS[k]}, {UNITS[k]}" for k in DESCRIBED],
        str(FIRST_YEAR): first.map(_num).values, str(LAST_YEAR): last.map(_num).values,
        "Изменение, %": ((last / first - 1) * 100).map(lambda v: f"{v:+.0f}").values,
    })
    skewed = stats.loc[stats["skew"].abs() > 1, "indicator"].map(LABELS.get).tolist()
    return f"""## 2. Описательная статистика

Все {len(panel)} наблюдений «регион × год». Денежные показатели — в ценах {BASE_YEAR} г.

{_table(shown)}

Сильная правая асимметрия (больше 1): {"; ".join(skewed)}. Несколько ресурсных и столичных
регионов резко выделяются, поэтому в моделях ВРП, инвестиции и доходы берутся в логарифмах.

**Медианный регион в начале и в конце периода**

{_table(change)}

![Распределения](figures/distributions.png)

![Диаграммы размаха по округам](figures/boxplots.png)

![Динамика социальных показателей](figures/trends.png)
"""


def _corr_section(corr: pd.DataFrame) -> str:
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(10)
    strongest = pd.DataFrame({
        "Показатель 1": [LABELS[a] for a, _ in top.index],
        "Показатель 2": [LABELS[b] for _, b in top.index],
        "r": top.map(lambda v: f"{v:+.2f}").values,
    })
    with_targets = pd.DataFrame({"Показатель": [LABELS[k] for k in DESCRIBED if k not in TARGETS]})
    for target in TARGETS:
        with_targets[LABELS[target]] = [f"{corr.loc[k, target]:+.2f}" for k in DESCRIBED if k not in TARGETS]
    return f"""## 3. Корреляционный анализ

Если считать корреляции по всем годам подряд, почти всё окажется связано со всем: за 24 года
вместе росли и доходы, и продолжительность жизни, и жилплощадь. Поэтому из каждого показателя
вычтено среднее по стране за этот год — остаются различия **между регионами**.
ВРП, инвестиции и доходы взяты в логарифмах.

![Матрица корреляций](figures/correlations.png)

**Десять самых сильных связей**

{_table(strongest, text_columns=2)}

**Связь с социальными показателями**

{_table(with_targets)}

Что видно:

- бедность тесно связана с доходами (r = {corr.loc["income_real", "poverty"]:+.2f}) и заметно слабее — с ВРП и инвестициями;
- продолжительность жизни с экономикой региона почти не связана
  (с доходами r = {corr.loc["income_real", "life_exp"]:+.2f}, с ВРП r = {corr.loc["gdp_real", "life_exp"]:+.2f});
- три экономических фактора сильно коррелируют между собой (ВРП и доходы: r = {corr.loc["gdp_real", "income_real"]:+.2f}),
  поэтому мультиколлинеарность проверена отдельно в разделе 4.

![Доходы и бедность](figures/income_poverty.png)
"""


def _models_section(coefs, summary, vif, tests) -> str:
    parts = ["""## 4. Регрессионный анализ

Зависимая переменная — социальный показатель, факторы — логарифмы ВРП, инвестиций и доходов
на душу населения. Оценены четыре модели:

| Модель | Спецификация |
|---|---|
| `naive` | факторы в текущих ценах + линейный тренд, все наблюдения в одной выборке |
| `real` | то же в реальных ценах |
| `fe` | реальные цены + фиксированные эффекты региона и года |
| `fe_controls` | то же + безработица, врачи, доля горожан, доля пожилых |

Метод — МНК; стандартные ошибки кластеризованы по регионам, потому что наблюдения одного региона
за разные годы не независимы. В скобках — стандартная ошибка; `***` p < 0,01, `**` p < 0,05, `*` p < 0,1.
"""]
    for target in TARGETS:
        rows = coefs[coefs["target"] == target]
        terms = [t for t in TERMS if t in set(rows["term"])]
        table = pd.DataFrame({"Переменная": [TERMS[t] for t in terms]})
        for spec in SPECS:
            cells = rows[rows["spec"] == spec].set_index("term")
            table[f"`{spec}`"] = [
                f"{cells.loc[t, 'coef']:.3f}{_stars(cells.loc[t, 'pvalue'])} ({cells.loc[t, 'se']:.3f})"
                if t in cells.index else "" for t in terms
            ]
        quality = summary[summary["target"] == target].set_index("spec")
        extra = pd.DataFrame(
            [["Наблюдений"] + [str(quality.loc[s, "n"]) for s in SPECS],
             ["R²"] + [f"{quality.loc[s, 'r2']:.3f}" for s in SPECS],
             ["R² внутри региона"] + [_num(quality.loc[s, "r2_within"], 3) for s in SPECS]],
            columns=table.columns)
        parts.append(f"### {LABELS[target]}, {UNITS[target]}\n\n{_table(pd.concat([table, extra]))}\n")

    effects = effect_of_10pct(coefs)
    best = effects[effects["spec"] == "fe_controls"]
    shown = pd.DataFrame({
        "Показатель": [LABELS[t] for t in best["target"]],
        "Фактор": [TERMS[t].replace("ln ", "") for t in best["term"]],
        "Эффект": best["effect"].map(lambda v: f"{v:+.3f}").values,
        "95 %-й интервал": [f"от {a:+.3f} до {b:+.3f}" for a, b in zip(best["ci_low"], best["ci_high"])],
        "p": best["pvalue"].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}").values,
    })
    vif_shown = pd.DataFrame({"Фактор": [LABELS[f] for f in vif["factor"]], "VIF": vif["vif"].map(_num)})
    tests_shown = pd.DataFrame({
        "Модель": [LABELS[t] for t in tests["target"]],
        "Ст. откл. остатков": tests["resid_std"].map(_num), "Асимметрия": tests["skew"].map(_num),
        "Эксцесс": tests["kurtosis"].map(_num), "Харке — Бера": tests["jarque_bera"].map(lambda v: _num(v, 1)),
        "p": tests["pvalue"].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"),
    })
    parts.append(f"""### Как читать коэффициенты

«R² внутри региона» — доля разброса, которую факторы объясняют **сверх** фиксированных эффектов,
то есть после того, как убраны постоянные различия регионов и общие для страны события года.
Обычный R² у моделей с эффектами близок к 1 просто потому, что в них сотня фиктивных переменных.

Коэффициент при логарифме показывает изменение показателя при росте фактора в e раз.
Нагляднее пересчитать его в эффект роста на 10 % (модель `fe_controls`):

{_table(shown, text_columns=2)}

![Эффекты факторов](figures/effects.png)

### Проверка мультиколлинеарности

{_table(vif_shown)}

Значения ниже 5 — коэффициенты оцениваются устойчиво, но разделить вклад ВРП и доходов
модель может лишь приблизительно.

### Остатки панельной модели

{_table(tests_shown)}

![Факт, прогноз и Q-Q графики](figures/diagnostics.png)

Остатки не нормальны: хвосты тяжелее, чем у нормального распределения (эксцесс больше 0).
На оценки коэффициентов это не влияет, а значимость при почти 2000 наблюдений и кластеризованных
ошибках опирается на асимптотику, а не на нормальность остатков.
""")
    return "\n".join(parts)


def _clusters_section(assignments, profiles, scores) -> str:
    cols = ["income_real", "gdp_real", "poverty", "life_exp", "housing", "unemployment", "doctors", "urban", "elderly"]
    table = pd.DataFrame({"Кластер": profiles["cluster"], "Регионов": profiles["regions"]})
    for col in cols:
        table[CLUSTER_COLUMNS[col]] = profiles[col].map(lambda v: _num(v, 1))
    lists = "\n".join(
        f"- **Кластер {cluster}** ({len(group)}): {', '.join(group['region'])}."
        for cluster, group in assignments.groupby("cluster"))
    silhouette = pd.DataFrame({"k": scores["k"], "Силуэт": scores["silhouette"].map(lambda v: f"{v:.3f}")})
    return f"""## 5. Типология регионов

Метод k-средних по {len(FEATURES)} стандартизованным показателям {LAST_YEAR} года (ВРП и доходы — в логарифмах).
Кластеры пронумерованы по убыванию среднего реального дохода.

{_table(table)}

{lists}

![Кластеры на плоскости главных компонент](figures/clusters.png)

**Выбор числа кластеров**

{_table(silhouette)}

По силуэту лучше k = 2, но такое деление лишь отделяет бедные республики от всех остальных.
Взято k = {N_CLUSTERS}: следующее по качеству и содержательно более полезное.
"""


def build_report(panel, russia, coefs, summary, vif, fits, assignments, profiles, scores) -> Path:
    stats, corr, tests = describe(panel), within_year_corr(panel), residual_tests(fits)
    quality = summary.set_index(["target", "spec"])
    income = effect_of_10pct(coefs).set_index(["target", "spec", "term"]).loc[("poverty", "fe_controls", "ln_income")]
    start, end = russia.iloc[0], russia.iloc[-1]

    text = f"""# Отчёт об анализе

Файл собирается автоматически командой `python -m regions`; все числа и графики получены из `data/raw`.

**Содержание:** [данные](#1-данные) · [описательная статистика](#2-описательная-статистика) ·
[корреляции](#3-корреляционный-анализ) · [регрессии](#4-регрессионный-анализ) ·
[типология](#5-типология-регионов) · [выводы](#6-выводы)

## 1. Данные

Панель: **{panel["region"].nunique()} региона, {FIRST_YEAR}–{LAST_YEAR} годы, {len(panel)} наблюдений**.
Источник — приложение к сборнику Росстата «Регионы России. Социально-экономические показатели» (2025),
подробности в [`data/SOURCES.md`](../data/SOURCES.md).

Денежные показатели пересчитаны в цены {BASE_YEAR} года через стоимость фиксированного набора
потребительских товаров и услуг в регионе. Это убирает и инфляцию, и разницу цен между регионами.
Зачем это нужно, видно на доходах России в целом: в текущих ценах они выросли
в {end["income"] / start["income"]:.1f} раза, в реальных — в {end["income_real"] / start["income_real"]:.1f}.

![Номинальные и реальные доходы](figures/nominal_vs_real.png)

![Неравенство регионов по доходам](figures/convergence.png)

{_describe_section(panel, stats)}
{_corr_section(corr)}
{_models_section(coefs, summary, vif, tests)}
{_clusters_section(assignments, profiles, scores)}
## 6. Выводы

1. **Бедность определяется доходами.** Рост реальных доходов на 10 % связан со снижением бедности
   на {abs(income["effect"]):.1f} п. п. (95 %-й интервал от {abs(income["ci_high"]):.1f} до {abs(income["ci_low"]):.1f}).
   ВРП и инвестиции при тех же доходах значимой связи не дают. Факторы объясняют
   {quality.loc[("poverty", "fe_controls"), "r2_within"] * 100:.0f} % разброса бедности внутри региона.
2. **Продолжительность жизни экономикой региона не объясняется.** R² наивной модели
   {quality.loc[("life_exp", "naive"), "r2"]:.2f} держится на общем тренде; внутри региона факторы объясняют
   {quality.loc[("life_exp", "fe"), "r2_within"] * 100:.0f} % разброса, с контрольными переменными —
   {quality.loc[("life_exp", "fe_controls"), "r2_within"] * 100:.0f} %.
3. **Обеспеченность жильём** растёт во всех регионах почти независимо от экономической динамики:
   внутри региона объясняется {quality.loc[("housing", "fe_controls"), "r2_within"] * 100:.0f} % разброса.
4. **Поправка на цены меняет картину:** без неё и рост, и разрыв между регионами выглядят намного больше.
5. **Регионы делятся на три устойчивых типа,** причём самый бедный кластер отличается
   самой высокой продолжительностью жизни.

Ограничения анализа перечислены в [README](../README.md).
"""
    REPORT_PATH.write_text(text, encoding="utf-8", newline="\n")
    stats.round(3).to_csv(REPORT_PATH.parent / "tables" / "descriptive_stats.csv", index=False)
    corr.round(3).to_csv(REPORT_PATH.parent / "tables" / "correlations.csv")
    tests.round(4).to_csv(REPORT_PATH.parent / "tables" / "residual_tests.csv", index=False)
    return REPORT_PATH
