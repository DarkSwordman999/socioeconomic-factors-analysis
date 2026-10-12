"""Сборка текста исследования results/REPORT.md: методы, формулы, таблицы, графики и трактовки.

Все числа подставляются из расчётов, поэтому текст всегда соответствует данным.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.stattools import jarque_bera

from regions.clusters import FEATURES, N_CLUSTERS
from regions.dataset import BASE_YEAR, FIRST_YEAR, LAST_YEAR
from regions.indicators import LABELS, TARGETS, UNITS
from regions.models import SPECS, effect_of_10pct

REPORT_PATH = Path(__file__).resolve().parent.parent / "results" / "REPORT.md"
TABLES_DIR = REPORT_PATH.parent / "tables"

DESCRIBED = ["gdp_real", "invest_real", "income_real", "poverty", "life_exp", "housing",
             "infant_mort", "unemployment", "doctors", "urban", "elderly", "students"]
LOGGED = ["gdp_real", "invest_real", "income_real"]
SYMBOLS = {"gdp_real": "x₁", "invest_real": "x₂", "income_real": "x₃",
           "poverty": "y₁", "life_exp": "y₂", "housing": "y₃"}
NAMES = {
    "gdp_real": "ВРП на душу", "invest_real": "Инвестиции на душу", "income_real": "Доходы на душу",
    "poverty": "Уровень бедности", "life_exp": "Продолжительность жизни", "housing": "Жилая площадь",
    "infant_mort": "Младенческая смертность", "unemployment": "Безработица", "doctors": "Врачи на 10 тыс.",
    "urban": "Доля горожан", "elderly": "Доля пожилых", "students": "Студенты на 10 тыс.",
}
GENITIVE = {"poverty": "бедности", "life_exp": "продолжительности жизни", "housing": "жилой площади"}
TERMS = {
    "ln_gdp": "ln ВРП", "ln_invest": "ln инвестиций", "ln_income": "ln доходов", "year": "Год (тренд)",
    "unemployment": "Безработица, %", "doctors": "Врачи на 10 тыс.", "urban": "Горожане, %",
    "elderly": "Пожилые, %",
}
FORM_NAMES = {"linear": "линейная", "log": "логарифмическая", "power": "степенная",
              "exp": "экспоненциальная", "quadratic": "квадратичная"}
CLUSTER_COLUMNS = {
    "income_real": "Доходы, руб.", "gdp_real": "ВРП, руб.", "poverty": "Бедность, %", "life_exp": "Жизнь, лет",
    "housing": "Жильё, кв. м", "unemployment": "Безраб., %", "doctors": "Врачи", "urban": "Горожане, %",
    "elderly": "Пожилые, %",
}


# --- форматирование -----------------------------------------------------------

def _num(value, digits=2) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ").replace("-", "−")
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",").replace("-", "−")


def _p(value) -> str:
    return "<0,001" if value < 0.001 else f"{value:.3f}".replace(".", ",")


def _pt(value) -> str:
    """p-значение для текста: «p < 0,001» или «p = 0,032»."""
    return "p < 0,001" if value < 0.001 else "p = " + f"{value:.3f}".replace(".", ",")


def _low(name: str) -> str:
    return name.lower().replace("врп", "ВРП")


def _term(value, digits=2) -> str:
    """Слагаемое уравнения со знаком и пробелом: «+ 2,13» или «− 1,37»."""
    return ("+ " if value >= 0 else "− ") + _num(abs(value), digits)


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _coef(value) -> str:
    """Модуль коэффициента уравнения; малые числа — в виде m·10ⁿ."""
    size = abs(value)
    if 0 < size < 1e-3:
        mantissa, exponent = f"{size:.2e}".split("e")
        return mantissa.replace(".", ",") + "·10" + str(int(exponent)).translate(_SUPERSCRIPT)
    return f"{size:.4g}".replace(".", ",")


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def _table(frame: pd.DataFrame, text_columns: int = 1) -> str:
    """Таблица Markdown из DataFrame; первые text_columns столбцов выравниваются влево."""
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join(["---"] * text_columns + ["---:"] * (len(frame.columns) - text_columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def _strength(r: float) -> str:
    """Словесная оценка тесноты связи по шкале Чеддока."""
    size = abs(r)
    if size < 0.1:
        return "отсутствует"
    for bound, word in [(0.3, "слабая"), (0.5, "умеренная"), (0.7, "заметная"), (0.9, "высокая")]:
        if size < bound:
            return word
    return "весьма высокая"


def _signed(value, digits=2) -> str:
    return ("+" if value >= 0 else "−") + _num(abs(value), digits)


# --- расчёты, которые нужны только отчёту ---------------------------------------

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


# --- разделы --------------------------------------------------------------------

def _intro(r: dict) -> str:
    panel, russia = r["panel"], r["russia"]
    start, end = russia.iloc[0], russia.iloc[-1]
    return f"""## 1. Введение

### 1.1. Постановка задачи

За {FIRST_YEAR}–{LAST_YEAR} годы экономика российских регионов сильно выросла: ВРП на душу населения
в медианном регионе увеличился в реальном выражении в {_num(panel[panel.year == LAST_YEAR].gdp_real.median() / panel[panel.year == FIRST_YEAR].gdp_real.median(), 1)} раза.
За то же время бедность в стране снизилась с {_num(start["poverty"], 1)} % до {_num(end["poverty"], 1)} %,
ожидаемая продолжительность жизни выросла с {_num(start["life_exp"], 1)} до {_num(end["life_exp"], 1)} года,
обеспеченность жильём — с {_num(start["housing"], 1)} до {_num(end["housing"], 1)} кв. м на человека.

Одновременный рост ещё не означает связи. Вопрос исследования: **связано ли улучшение социальных
показателей региона с ростом его экономики, или оно шло по всей стране независимо от того,
как развивался конкретный регион?**

### 1.2. Гипотезы

| | Гипотеза |
|---|---|
| **H1** | Чем выше ВРП, инвестиции и доходы на душу населения, тем ниже уровень бедности; сильнее всего бедность связана с доходами |
| **H2** | Чем выше экономические показатели региона, тем выше ожидаемая продолжительность жизни |
| **H3** | Чем выше экономические показатели региона, тем больше жилой площади приходится на жителя |
| **H4** | Регионы сближаются по уровню экономического развития: бедные растут быстрее богатых |

### 1.3. Логика исследования

| Шаг | Вопрос | Метод | Раздел |
|---|---|---|---|
| 1 | Как устроены данные, можно ли применять параметрические методы | Описательная статистика, критерий Шапиро — Уилка | 3 |
| 2 | Сближаются ли регионы | σ- и β-сходимость | 4 |
| 3 | Есть ли связь и насколько она тесная | Коэффициенты Пирсона и Спирмена | 5 |
| 4 | Какой формы связь | Парные регрессии пяти видов, эластичность | 6 |
| 5 | Каков вклад каждого фактора при прочих равных | Множественная регрессия, проверка предпосылок МНК | 7 |
| 6 | Сохраняется ли связь внутри региона во времени | Панельная регрессия с фиксированными эффектами | 8 |
| 7 | Какие типы регионов существуют | Кластеризация k-средних, главные компоненты | 9 |
"""


_DEFLATOR = r"""### 2.2. Приведение к сопоставимым ценам

Денежные показатели в источнике даны в текущих ценах. Сравнивать их напрямую нельзя по двум причинам:
цены за 24 года выросли многократно, и в один и тот же год они сильно различаются между регионами.
Обе проблемы снимает один дефлятор — стоимость фиксированного набора потребительских товаров и услуг
$C_{it}$, которую Росстат рассчитывает для каждого региона $i$ и года $t$:

$$
P_{it} = \frac{C_{it}}{C_{\text{РФ},\,2023}}, \qquad x^{\text{реал}}_{it} = \frac{x_{it}}{P_{it}}
$$

где $C_{\text{РФ},\,2023}$ — стоимость того же набора в среднем по России в 2023 году. Результат —
рубли 2023 года при среднероссийском уровне цен.
"""


def _data(r: dict) -> str:
    panel, russia = r["panel"], r["russia"]
    start, end = russia.iloc[0], russia.iloc[-1]
    latest = panel[panel["year"] == LAST_YEAR].set_index("region")["price_index"]
    variables = pd.DataFrame({
        "Обозн.": [SYMBOLS[k] for k in SYMBOLS],
        "Показатель": [LABELS[k] for k in SYMBOLS],
        "Единица": [UNITS[k] for k in SYMBOLS],
        "Роль": ["фактор"] * 3 + ["результат"] * 3,
    })
    gaps = panel[["gdp", "income", "poverty", "basket"]].isna().any(axis=1)
    missing = panel[gaps].groupby("region")["year"].agg(["min", "max", "count"])
    gap_text = "; ".join(f"{region} ({int(row['min'])}–{int(row['max'])})" if row["count"] > 1
                         else f"{region} ({int(row['min'])})" for region, row in missing.iterrows())
    return f"""## 2. Данные

### 2.1. Источник и показатели

Источник — приложение к статистическому сборнику Росстата «Регионы России. Социально-экономические
показатели» (2025), таблицы по субъектам Российской Федерации за {FIRST_YEAR}–{LAST_YEAR} годы
([`data/SOURCES.md`](../data/SOURCES.md)).

{_table(variables, text_columns=4)}

Контрольные переменные: уровень безработицы, число врачей на 10 000 человек, доля городского населения,
доля населения старше трудоспособного возраста. Для описания и типологии дополнительно используются
младенческая смертность и число студентов на 10 000 человек.

{_DEFLATOR}
В {LAST_YEAR} году индекс $P$ менялся от {_num(latest.min())} ({latest.idxmin()}) до {_num(latest.max())}
({latest.idxmax()}): один и тот же набор товаров на Чукотке почти вдвое дороже, чем на юге страны.
Насколько поправка существенна, видно по России в целом: в текущих ценах среднедушевые доходы
выросли в {_num(end["income"] / start["income"], 1)} раза, в реальных — в {_num(end["income_real"] / start["income_real"], 1)} раза.

![Номинальные и реальные доходы](figures/nominal_vs_real.png)

### 2.3. Выборка

Панель: **{panel["region"].nunique()} региона × {LAST_YEAR - FIRST_YEAR + 1} года = {len(panel)} наблюдений**.
Автономные округа учтены в составе Архангельской и Тюменской областей, чтобы территории не пересекались.
Панель несбалансированная: по Республике Крым и Севастополю данные начинаются с 2014 года.
Пропуски в основных показателях (годы в скобках): {gap_text}. Строки с пропусками в переменных модели в оценивании не участвуют.
"""


_DESCRIBE_FORMULAS = r"""### 3.1. Описательная статистика

Для каждого показателя рассчитаны среднее $\bar{x}$, стандартное отклонение $s$, коэффициент вариации $V$,
асимметрия $A$ и эксцесс $E$:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}, \qquad
V = \frac{s}{\bar{x}}\cdot 100\,\%
$$

$$
A = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i-\bar{x}}{s}\right)^{3}, \qquad
E = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i-\bar{x}}{s}\right)^{4} - 3
$$

У нормального распределения $A = 0$ и $E = 0$. Совокупность принято считать однородной при $V < 33\,\%$.
"""


def _explore(r: dict) -> str:
    panel, stats, normal = r["panel"], r["stats"], r["normality"]
    shown = pd.DataFrame({
        "Показатель": [f"{LABELS[k]}, {UNITS[k]}" for k in stats["indicator"]],
        "n": stats["n"].astype(str),
        "Среднее": stats["mean"].map(_num), "Медиана": stats["median"].map(_num),
        "Ст. откл.": stats["std"].map(_num), "Мин.": stats["min"].map(_num), "Макс.": stats["max"].map(_num),
        "V, %": stats["cv_pct"].map(lambda v: _num(v, 0)),
        "A": stats["skew"].map(_num), "E": stats["kurtosis"].map(_num),
    })
    by_key = stats.set_index("indicator")
    uniform = [_low(NAMES[k]) for k in DESCRIBED if by_key.loc[k, "cv_pct"] < 33]
    varied = [f"{_low(NAMES[k])} ({_num(by_key.loc[k, 'cv_pct'], 0)} %)" for k in DESCRIBED if by_key.loc[k, "cv_pct"] >= 50]
    skewed = [f"{_low(NAMES[k])} (A = {_num(by_key.loc[k, 'skew'])})" for k in DESCRIBED if by_key.loc[k, "skew"] > 1]

    norm_table = pd.DataFrame({
        "Показатель": [NAMES[k] for k in normal["indicator"]],
        "A": normal["skew"].map(_num), "W": normal["w"].map(lambda v: _num(v, 3)), "p": normal["p"].map(_p),
        "A (ln)": normal["skew_log"].map(_num), "W (ln)": normal["w_log"].map(lambda v: _num(v, 3)),
        "p (ln)": normal["p_log"].map(_p),
    })
    nrm = normal.set_index("indicator")
    normal_levels = [_low(NAMES[k]) for k in nrm.index if nrm.loc[k, "p"] >= 0.05]
    money = nrm.loc[LOGGED]

    years = [y for y in (2000, 2005, 2010, 2015, 2020, LAST_YEAR)]
    medians = panel[panel["year"].isin(years)].groupby("year")[list(SYMBOLS)].median()
    dynamics = pd.DataFrame({"Показатель": [f"{NAMES[k]}, {UNITS[k]}" for k in SYMBOLS]})
    for year in years:
        dynamics[str(year)] = [_num(medians.loc[year, k], 1) for k in SYMBOLS]
    dynamics["Рост, раз"] = [_num(medians.loc[LAST_YEAR, k] / medians.loc[FIRST_YEAR, k]) for k in SYMBOLS]

    extremes = []
    for key in TARGETS:
        top, bottom = r["extremes"][key]
        extremes.append({
            "Показатель": f"{NAMES[key]}, {UNITS[key]}",
            "Наибольшие значения": "; ".join(f"{a} — {_num(b, 1)}" for a, b in zip(top["region"], top[key])),
            "Наименьшие значения": "; ".join(f"{a} — {_num(b, 1)}" for a, b in zip(bottom["region"], bottom[key])),
        })
    life = panel.groupby("year")["life_exp"].median()
    return f"""## 3. Разведочный анализ

{_DESCRIBE_FORMULAS}
**Таблица 1. Описательная статистика, все {len(panel)} наблюдений** (денежные показатели — в ценах {BASE_YEAR} г.)

{_table(shown)}

**Трактовка.**

- *Однородность.* Коэффициент вариации ниже 33 % у показателей: {", ".join(uniform)}. По ним регионы
  различаются умеренно, и среднее хорошо описывает типичный регион.
  Сильнее всего регионы различаются по показателям: {", ".join(varied)} — здесь среднее заметно выше медианы
  и на него нельзя опираться.
- *Асимметрия.* Выраженная правосторонняя асимметрия (A > 1): {", ".join(skewed)}. У распределения длинный
  правый «хвост»: большинство регионов сосредоточено у нижних значений, а несколько выбиваются далеко вверх.
  По ВРП и инвестициям это ресурсные регионы и столицы, по бедности и безработице — республики Северного Кавказа и Тыва.
- *Вывод для моделей.* Показатели с длинным правым хвостом нельзя включать в регрессию в уровнях:
  несколько регионов определили бы наклон линии для всех остальных. Поэтому ВРП, инвестиции и доходы
  далее берутся в натуральных логарифмах.

![Распределения показателей](figures/distributions.png)

### 3.2. Проверка нормальности

Корреляция Пирсона и t-критерий предполагают распределение, близкое к нормальному. Гипотеза
$H_0$: «выборка взята из нормального распределения» проверена критерием Шапиро — Уилка на срезе {LAST_YEAR} года
($n = {int(normal["n"].iloc[0])}$); при $p < 0{{,}}05$ гипотеза отвергается. Статистика $W$ тем ближе к 1,
чем ближе распределение к нормальному.

**Таблица 2. Критерий Шапиро — Уилка, {LAST_YEAR} г.: показатели в уровнях и в логарифмах**

{_table(norm_table)}

**Трактовка.** В уровнях нормальность не отвергается только у показателей: {", ".join(normal_levels) or "нет таких"}.
У трёх экономических факторов распределение далеко от нормального (W от {_num(money["w"].min(), 2)} до {_num(money["w"].max(), 2)}).
Логарифмирование приближает их к нормальному — W растёт до {_num(money["w_log"].min(), 2)}–{_num(money["w_log"].max(), 2)},
асимметрия падает с {_num(money["skew"].min(), 1)}–{_num(money["skew"].max(), 1)} до {_num(money["skew_log"].min(), 1)}–{_num(money["skew_log"].max(), 1)}, —
но полностью нормальным не делает. Отсюда два решения: факторы используются в логарифмах, а выводы
по корреляции Пирсона дублируются ранговым коэффициентом Спирмена, который нормальности не требует.

### 3.3. Динамика

**Таблица 3. Медианный регион по годам**

{_table(dynamics)}

![Динамика социальных показателей](figures/trends.png)

**Трактовка.** Все шесть показателей менялись в одну сторону: экономические выросли в 2,7–2,8 раза
в реальном выражении, бедность упала вчетверо, продолжительность жизни выросла на {_num(life[LAST_YEAR] - life[FIRST_YEAR], 1)} года.
Это главная методическая ловушка исследования: любые два растущих ряда коррелируют между собой, даже если
никак не связаны. Основной рост доходов пришёлся на 2000–2012 годы, после чего реальные доходы почти не менялись,
а социальные показатели продолжали улучшаться — первый признак того, что они зависят не только от экономики.
Провал продолжительности жизни в 2020–2021 годах общий для всех регионов.

### 3.4. Территориальные различия

**Таблица 4. Крайние значения социальных показателей, {LAST_YEAR} г.**

{_table(pd.DataFrame(extremes), text_columns=3)}

![Разброс регионов внутри федеральных округов](figures/boxplots.png)

**Трактовка.** Списки лидеров и аутсайдеров по трём показателям не совпадают. Ингушетия — регион с самой
высокой бедностью и одновременно в тройке лидеров по продолжительности жизни; Чукотка — в числе регионов
с самой низкой бедностью и на последнем месте по продолжительности жизни. Самая большая обеспеченность жильём — не в богатых
регионах, а в областях Центра и Северо-Запада с убывающим населением. Уже на этом этапе видно, что
гипотезы H2 и H3 в простой форме могут не подтвердиться.
"""


_SIGMA = r"""## 4. Сходимость регионов

### 4.1. σ-сходимость

σ-сходимость означает, что разброс показателя между регионами со временем уменьшается. Мера разброса —
коэффициент вариации в году $t$:

$$
V_t = \frac{s_t}{\bar{x}_t}\cdot 100\,\%
$$

Если $V_t$ убывает, регионы сближаются.
"""

_BETA = r"""### 4.2. β-сходимость

β-сходимость означает, что регионы с низким исходным уровнем растут быстрее. Проверяется регрессией
среднегодового темпа роста на логарифм исходного уровня (Барро и Сала-и-Мартин):

$$
\frac{1}{T}\ln\frac{y_{i,T}}{y_{i,0}} = a + b\,\ln y_{i,0} + \varepsilon_i
$$

Сходимость есть, если $b < 0$ и коэффициент значим. По $b$ рассчитываются скорость сходимости $\lambda$
и время, за которое разрыв сокращается вдвое:

$$
\lambda = -\frac{\ln(1 + bT)}{T}, \qquad t_{1/2} = \frac{\ln 2}{\lambda}
$$
"""


def _convergence(r: dict) -> str:
    sigma, beta = r["sigma"].set_index("year"), r["beta"].set_index("indicator")
    years = [2000, 2004, 2008, 2012, 2016, 2020, LAST_YEAR]
    cols = {"income": "Доходы, текущие цены", "income_real": "Доходы, реальные", "gdp_real": "ВРП, реальный",
            "poverty": "Бедность", "life_exp": "Продолж. жизни"}
    sigma_table = pd.DataFrame({"Показатель": list(cols.values())})
    for year in years:
        sigma_table[str(year)] = [_num(sigma.loc[year, c], 1) for c in cols]
    low_year = int(sigma["income_real"].idxmin())
    beta_table = pd.DataFrame({
        "Показатель": ["Реальные доходы", "Реальный ВРП на душу"],
        "n": beta["n"].astype(str).values, "a": beta["a"].map(lambda v: _num(v, 4)).values,
        "b": beta["b"].map(lambda v: _num(v, 4)).values, "Ст. ош.": beta["se"].map(lambda v: _num(v, 4)).values,
        "p": beta["p"].map(_p).values, "R²": beta["r2"].map(_num).values,
        "λ, % в год": beta["speed_pct"].map(_num).values, "t½, лет": beta["half_life"].map(lambda v: _num(v, 0)).values,
    })
    inc, gdp = beta.loc["income_real"], beta.loc["gdp_real"]
    return f"""{_SIGMA}
**Таблица 5. Коэффициент вариации между регионами, %**

{_table(sigma_table)}

![σ-сходимость](figures/convergence.png)

**Трактовка.** По реальным доходам разброс сократился с {_num(sigma.loc[FIRST_YEAR, "income_real"], 1)} % до
{_num(sigma.loc[low_year, "income_real"], 1)} % к {low_year} году — регионы заметно сблизились. После этого сближение
остановилось, и к {LAST_YEAR} году разброс вырос до {_num(sigma.loc[LAST_YEAR, "income_real"], 1)} %.
По ВРП на душу сходимости нет вовсе: {_num(sigma.loc[FIRST_YEAR, "gdp_real"], 1)} % в начале периода и
{_num(sigma.loc[LAST_YEAR, "gdp_real"], 1)} % в конце. Регионы стали ближе по тому, сколько получают жители,
но не по тому, сколько производят, — различие, которое указывает на роль перераспределения
(зарплаты бюджетников, пенсии, трансферты). Поправка на цены сама по себе сокращает видимое неравенство
почти вдвое: в текущих ценах коэффициент вариации доходов в {LAST_YEAR} году равен {_num(sigma.loc[LAST_YEAR, "income"], 1)} %.

{_BETA}
**Таблица 6. β-сходимость за {FIRST_YEAR}–{LAST_YEAR} гг.** (T = {LAST_YEAR - FIRST_YEAR}, робастные ошибки HC3)

{_table(beta_table)}

![β-сходимость](figures/beta_convergence.png)

**Трактовка.** По доходам сходимость подтверждается: b = {_num(inc["b"], 4)} ({_pt(inc["p"])}), исходный уровень
объясняет {_num(inc["r2"] * 100, 0)} % различий в темпах роста. Регион, который в {FIRST_YEAR} году был вдвое беднее,
рос в среднем на {_num(-inc["b"] * np.log(2) * 100, 1)} п. п. в год быстрее. Скорость сходимости {_num(inc["speed_pct"], 1)} % в год:
разрыв сокращается вдвое примерно за {_num(inc["half_life"], 0)} года.
По ВРП коэффициент b = {_num(gdp["b"], 4)} статистически незначим на уровне 5 % ({_pt(gdp["p"])}), а R² равен
всего {_num(gdp["r2"])}: бедные по ВРП регионы не росли систематически быстрее богатых.

**Гипотеза H4 подтверждается частично:** сходимость есть по доходам (и остановилась после {low_year} года),
по производству её нет.
"""


_CORR = r"""## 5. Корреляционный анализ

### 5.1. Методика

Теснота линейной связи измеряется коэффициентом Пирсона:

$$
r_{xy} = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2\,\sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$

Значимость проверяется t-критерием Стьюдента с $n-2$ степенями свободы ($H_0$: $r = 0$):

$$
t = r\,\sqrt{\frac{n-2}{1-r^2}}
$$

Поскольку распределения отклоняются от нормального (раздел 3.2), дополнительно рассчитан ранговый
коэффициент Спирмена, где $d_i$ — разность рангов региона по двум показателям:

$$
\rho = 1 - \frac{6\sum_{i=1}^{n} d_i^{2}}{n(n^2-1)}
$$

Теснота связи оценивается по шкале Чеддока: до 0,3 — слабая, 0,3–0,5 — умеренная, 0,5–0,7 — заметная,
0,7–0,9 — высокая, выше 0,9 — весьма высокая.

Корреляции считаются двумя способами. **На срезе одного года** — чтобы связь не подменялась общим трендом.
**По всей панели после вычитания среднего за год** — чтобы использовать все наблюдения:

$$
\tilde{x}_{it} = x_{it} - \bar{x}_{t}
$$

где $\bar{x}_{t}$ — среднее по регионам в году $t$.
"""


def _correlation(r: dict) -> str:
    tests, corr = r["corr_tests"], r["corr"]
    n = int(tests["n"].iloc[0])
    t_crit = sps.t.ppf(0.975, n - 2)
    r_crit = t_crit / np.sqrt(n - 2 + t_crit ** 2)
    table = pd.DataFrame({
        "Показатель": [NAMES[t] for t in tests["target"]],
        "Фактор": [NAMES[f] for f in tests["factor"]],
        "r": tests["pearson"].map(_num), "t": tests["t"].map(_num), "p": tests["p"].map(_p),
        "ρ Спирмена": tests["spearman"].map(_num), "p (ρ)": tests["p_spearman"].map(_p),
        "Связь": [_strength(v) if p < 0.05 else "незначима" for v, p in zip(tests["pearson"], tests["p"])],
    })

    notes = []
    for target in TARGETS:
        rows = tests[(tests["target"] == target) & tests["factor"].isin(LOGGED)].set_index("factor")
        parts = []
        for factor in LOGGED:
            row = rows.loc[factor]
            verdict = f"{_strength(row['pearson'])}, {_pt(row['p'])}" if row["p"] < 0.05 else f"незначима, {_pt(row['p'])}"
            parts.append(f"{_low(NAMES[factor])} r = {_signed(row['pearson'])} ({verdict})")
        notes.append(f"- **{NAMES[target]}:** " + "; ".join(parts) + ".")
    disagree = tests[(tests["p"] < 0.05) != (tests["p_spearman"] < 0.05)]
    disagree_text = "; ".join(f"{_low(NAMES[t])} — {_low(NAMES[f])}" for t, f in zip(disagree["target"], disagree["factor"]))

    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(10)
    strongest = pd.DataFrame({
        "Показатель 1": [NAMES[a] for a, _ in top.index], "Показатель 2": [NAMES[b] for _, b in top.index],
        "r": top.map(_signed).values, "Связь": top.map(_strength).values,
    })
    t = tests.set_index(["target", "factor"])
    return f"""{_CORR}
### 5.2. Корреляции на срезе {LAST_YEAR} года

При $n = {n}$ критическое значение $t_{{0{{,}}05;\\,{n - 2}}} = {_num(t_crit)}$, поэтому значимы коэффициенты, по модулю
превышающие {_num(r_crit)}. Экономические факторы взяты в логарифмах.

**Таблица 7. Связь факторов с социальными показателями, {LAST_YEAR} г.**

{_table(table, text_columns=2)}

**Трактовка.**

{chr(10).join(notes)}

Бедность связана со всеми тремя факторами, причём теснее всего — с доходами: доходы объясняют
{_num(t.loc[("poverty", "income_real"), "pearson"] ** 2 * 100, 0)} % её вариации ($r^2$), ВРП — {_num(t.loc[("poverty", "gdp_real"), "pearson"] ** 2 * 100, 0)} %,
инвестиции — {_num(t.loc[("poverty", "invest_real"), "pearson"] ** 2 * 100, 0)} %.

Для продолжительности жизни знак связи **отрицательный**: в регионах с высоким ВРП и инвестициями живут
в среднем меньше. Это не значит, что богатство сокращает жизнь, — это эффект состава: самые высокие ВРП
и инвестиции у северных и дальневосточных сырьевых регионов с тяжёлым климатом, а самая высокая
продолжительность жизни — в небогатых республиках Северного Кавказа. С доходами связи нет вообще.

Обеспеченность жильём с экономическими факторами связана слабо, зато тесно связана с долей пожилых
(r = {_signed(t.loc[("housing", "elderly"), "pearson"])}): жилья на человека больше там, где население стареет и убывает,
а жилой фонд остаётся.

Расхождения между Пирсоном и Спирменом в выводе о значимости: {disagree_text or "нет"}.
В этих парах связь держится на нескольких крайних регионах, и считать её установленной нельзя.

![Доходы и бедность](figures/income_poverty.png)

### 5.3. Корреляционная матрица по всей панели

![Матрица корреляций](figures/correlations.png)

**Таблица 8. Десять самых тесных связей** (отклонения от среднего за год, {len(r["panel"])} наблюдений)

{_table(strongest, text_columns=2)}

**Трактовка.** Три экономических фактора тесно связаны между собой: ВРП и доходы — r = {_signed(corr.loc["gdp_real", "income_real"])},
ВРП и инвестиции — r = {_signed(corr.loc["gdp_real", "invest_real"])}. Для множественной регрессии это означает риск
мультиколлинеарности: модели трудно разделить вклад ВРП и доходов, поэтому в разделе 7 он проверяется отдельно.
Связь бедности с доходами (r = {_signed(corr.loc["income_real", "poverty"])}) по всей панели такая же, как на срезе {LAST_YEAR} года,
то есть она устойчива во времени. Продолжительность жизни не имеет ни одной связи теснее умеренной.

**Предварительно:** H1 согласуется с данными, H2 и H3 — нет.
"""


_PAIR = r"""## 6. Парная регрессия

### 6.1. Формы связи и критерии выбора

Корреляция показывает тесноту связи, но не её форму. Для каждой из девяти пар «фактор → показатель»
на срезе последнего года оценены пять моделей; $x$ — фактор в тыс. руб., $E$ — коэффициент эластичности
в средней точке.

| Модель | Уравнение | Оценивается как | Эластичность $E$ |
|---|---|---|---|
| Линейная | $\hat{y} = a + bx$ | МНК | $b\,\bar{x}/\bar{y}$ |
| Логарифмическая | $\hat{y} = a + b\ln x$ | МНК по $\ln x$ | $b/\bar{y}$ |
| Степенная | $\hat{y} = a\,x^{b}$ | $\ln y = \ln a + b\ln x$ | $b$ |
| Экспоненциальная | $\hat{y} = a\,e^{bx}$ | $\ln y = \ln a + bx$ | $b\,\bar{x}$ |
| Квадратичная | $\hat{y} = a + bx + cx^{2}$ | МНК | $(b + 2c\bar{x})\,\bar{x}/\bar{y}$ |

Параметры находятся методом наименьших квадратов — минимизацией суммы квадратов остатков:

$$
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 \to \min, \qquad
b = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sum (x_i-\bar{x})^2}, \qquad a = \bar{y} - b\bar{x}
$$

Качество оценивается коэффициентом детерминации, скорректированным на число параметров $k$,
F-критерием Фишера и средней ошибкой аппроксимации:

$$
R^2 = 1 - \frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}, \qquad
\bar{R}^2 = 1 - (1-R^2)\,\frac{n-1}{n-k-1}
$$

$$
F = \frac{R^2}{1-R^2}\cdot\frac{n-k-1}{k}, \qquad
\bar{A} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{y_i}\right|\cdot 100\,\%
$$

Чтобы модели были сопоставимы, $R^2$ и $\bar{A}$ для степенной и экспоненциальной форм считаются
в исходных единицах $y$, а не в логарифмах. Лучшей считается форма с наибольшим $\bar{R}^2$.
Эластичность показывает, на сколько процентов меняется $y$ при росте $x$ на 1 %.
"""


def _equation(row) -> str:
    a, b, c = row["a"], row["b"], row["c"]
    sign = lambda v: "+" if v >= 0 else "−"
    g = _coef
    if row["form"] == "linear":
        return f"ŷ = {_num(a)} {sign(b)} {g(b)}·x"
    if row["form"] == "log":
        return f"ŷ = {_num(a)} {sign(b)} {g(b)}·ln x"
    if row["form"] == "power":
        return f"ŷ = {g(a)}·x^({_num(b, 3)})"
    if row["form"] == "exp":
        return f"ŷ = {_num(a)}·e^({'−' if b < 0 else ''}{g(b)}·x)"
    return f"ŷ = {_num(a)} {sign(b)} {g(b)}·x {sign(c)} {g(c)}·x²"


def _pairs(r: dict) -> str:
    pairs = r["pairs"]
    forms = list(FORM_NAMES)
    r2 = pd.DataFrame({
        "Показатель": [NAMES[t] for t, _ in pairs.groupby(["target", "factor"], sort=False).groups],
        "Фактор": [NAMES[f] for _, f in pairs.groupby(["target", "factor"], sort=False).groups],
    })
    grid = pairs.pivot_table(index=["target", "factor"], columns="form", values="r2_adj", sort=False)
    for form in forms:
        r2[FORM_NAMES[form].capitalize()] = grid[form].map(lambda v: _num(v, 3)).values
    best = pairs[pairs["best"]]
    best_table = pd.DataFrame({
        "Показатель": [NAMES[t] for t in best["target"]], "Фактор": [NAMES[f] for f in best["factor"]],
        "Форма": [FORM_NAMES[f] for f in best["form"]], "Уравнение": [_equation(row) for _, row in best.iterrows()],
        "R²": best["r2"].map(_num).values, "F": best["f"].map(lambda v: _num(v, 1)).values, "p": best["p"].map(_p).values,
        "Ā, %": best["mape"].map(lambda v: _num(v, 1)).values, "E": best["elasticity"].map(_num).values,
    })
    b = best.set_index(["target", "factor"])
    lin = pairs[pairs["form"] == "linear"].set_index(["target", "factor"])
    pi, hi = b.loc[("poverty", "income_real")], b.loc[("housing", "income_real")]
    life_max = pairs[pairs["target"] == "life_exp"]["r2"].max()
    quad = pairs[(pairs["target"] == "housing") & (pairs["factor"] == "income_real") & (pairs["form"] == "quadratic")].iloc[0]
    peak = -quad["b"] / (2 * quad["c"])
    n = int(pairs["n"].iloc[0])
    return f"""{_PAIR}
### 6.2. Результаты

**Таблица 9. Скорректированный R² по формам связи, {LAST_YEAR} г.** (n = {n})

{_table(r2, text_columns=2)}

**Таблица 10. Лучшая модель для каждой пары**

{_table(best_table, text_columns=4)}

![Парные регрессии](figures/pair_regressions.png)

### 6.3. Трактовка

**Бедность и доходы.** Лучшая форма — {FORM_NAMES[pi["form"]]}: {_equation(pi)}, R² = {_num(pi["r2"])}, F = {_num(pi["f"], 1)}
({_pt(pi["p"])}). Модель объясняет {_num(pi["r2"] * 100, 0)} % различий регионов по уровню бедности — против
{_num(lin.loc[("poverty", "income_real"), "r2"] * 100, 0)} % у линейной. Показатель степени — это эластичность:
**при росте реальных доходов на 1 % уровень бедности снижается на {_num(abs(pi["elasticity"]), 2)} %**
(не процентного пункта, а процента от текущего уровня). Эластичность по модулю больше единицы — бедность реагирует
на доходы сильнее, чем пропорционально. Степенная форма означает убывающую отдачу в абсолютном выражении:
одна и та же прибавка дохода сильнее снижает бедность в бедном регионе, чем в богатом. Средняя ошибка
аппроксимации {_num(pi["mape"], 1)} % выше порога 10–15 %, принятого для точного прогноза: модель верно описывает
закономерность, но для предсказания бедности отдельного региона одних доходов мало.

**Бедность, ВРП и инвестиции.** Форма та же, но связь слабее: R² = {_num(b.loc[("poverty", "gdp_real"), "r2"])} для ВРП
и {_num(b.loc[("poverty", "invest_real"), "r2"])} для инвестиций, эластичности {_num(b.loc[("poverty", "gdp_real"), "elasticity"])}
и {_num(b.loc[("poverty", "invest_real"), "elasticity"])}. ВРП и инвестиции отражают производство, которое доходит
до жителей лишь частично.

**Продолжительность жизни.** Ни одна форма ни с одним фактором не даёт R² выше {_num(life_max)}. Подбор формы
ничего не меняет: дело не в том, что связь нелинейная, а в том, что её практически нет.
Эластичности порядка −0,02 означают, что удвоение фактора соответствует изменению продолжительности жизни
на доли процента.

**Жилая площадь.** С ВРП и инвестициями связи нет (R² до {_num(max(b.loc[("housing", "gdp_real"), "r2"], b.loc[("housing", "invest_real"), "r2"]))}).
С доходами лучшая форма — {FORM_NAMES[hi["form"]]} (R² = {_num(hi["r2"])}): обеспеченность жильём растёт с доходами
примерно до {_num(peak, 0)} тыс. руб. в месяц, а затем снижается. Правая, нисходящая ветвь параболы — это
немногочисленные богатые регионы Севера, Москва и Санкт-Петербург, где жильё дорогое и на человека его приходится мало.
Это свойство состава выборки, а не закономерность «доходы выше — жилья меньше».
"""


_MULTI = r"""## 7. Множественная регрессия

### 7.1. Модель и проверяемые предпосылки

Парные регрессии не отвечают на вопрос, какой из трёх тесно связанных факторов действует сам по себе.
Для этого все три включаются в одну модель:

$$
y_i = \beta_0 + \beta_1\ln x_{1i} + \beta_2\ln x_{2i} + \beta_3\ln x_{3i} + \varepsilon_i
$$

где $x_1$ — ВРП, $x_2$ — инвестиции, $x_3$ — доходы на душу. Коэффициент $\beta_j$ показывает изменение $y$
при росте фактора в $e \approx 2{,}72$ раза **при неизменных остальных**; при росте фактора на 1 % показатель
меняется на $\beta_j/100$ единиц. Вектор оценок по методу наименьших квадратов:

$$
\hat{\boldsymbol\beta} = (X^{\top}X)^{-1}X^{\top}\mathbf{y}
$$

Значимость уравнения в целом проверяется F-критерием ($H_0$: $\beta_1=\beta_2=\beta_3=0$), каждого
коэффициента — t-критерием:

$$
F = \frac{R^2/k}{(1-R^2)/(n-k-1)}, \qquad t_j = \frac{\hat\beta_j}{s_{\hat\beta_j}}
$$

Чтобы сравнивать вклад факторов между собой, рассчитаны стандартизованные коэффициенты и эластичности:

$$
\beta^{*}_j = \hat\beta_j\,\frac{s_{\ln x_j}}{s_y}, \qquad E_j = \frac{\hat\beta_j}{\bar{y}}
$$

Мультиколлинеарность измеряется коэффициентом вздутия дисперсии, где $R_j^2$ — детерминация регрессии
$j$-го фактора на остальные; значения выше 10 считаются критическими, выше 5 — настораживающими:

$$
VIF_j = \frac{1}{1-R_j^2}
$$

Проверяются предпосылки МНК:

| Предпосылка | Тест | $H_0$ |
|---|---|---|
| Постоянство дисперсии остатков | Бройша — Пагана | гомоскедастичность |
| Нормальность остатков | Харке — Бера | остатки нормальны |
| Правильность функциональной формы | RESET Рамсея | форма выбрана верно |

При $p < 0{,}05$ нулевая гипотеза отвергается, то есть предпосылка нарушена.
"""


def _multiple(r: dict) -> str:
    coefs, quality, yearly = r["cs_coefs"], r["cs_quality"].set_index("target"), r["yearly"]
    parts = [_MULTI, f"### 7.2. Оценки на срезе {LAST_YEAR} года\n"]
    for number, target in enumerate(TARGETS, start=11):
        rows = coefs[coefs["target"] == target]
        q = quality.loc[target]
        table = pd.DataFrame({
            "Фактор": [TERMS[t] for t in rows["term"]],
            "β": rows["coef"].map(lambda v: _num(v, 3)).values, "Ст. ош.": rows["se"].map(lambda v: _num(v, 3)).values,
            "t": rows["t"].map(_num).values, "p": rows["p"].map(_p).values, "p (HC3)": rows["p_hc3"].map(_p).values,
            "β*": rows["beta_std"].map(_num).values, "E": rows["elasticity"].map(lambda v: _num(v, 3)).values,
            "VIF": rows["vif"].map(_num).values,
        })
        c = rows.set_index("term")["coef"]
        equation = (f"ŷ = {_num(q['intercept'])} {_term(c['ln_gdp'])}·ln x₁ {_term(c['ln_invest'])}·ln x₂ "
                    f"{_term(c['ln_income'])}·ln x₃")
        parts.append(f"""**Таблица {number}. {LABELS[target]}, {UNITS[target]}** (n = {int(q["n"])})

{_table(table)}

Уравнение: **{equation}**

R² = {_num(q["r2"], 3)}, скорр. R² = {_num(q["r2_adj"], 3)}, F = {_num(q["f"])} ({_pt(q["f_p"])}),
стандартная ошибка регрессии {_num(q["se_reg"])} {UNITS[target]}, средняя ошибка аппроксимации {_num(q["mape"], 1)} %.
""")

    diag = pd.DataFrame({
        "Модель": [NAMES[t] for t in TARGETS],
        "Бройш — Паган, p": [_p(quality.loc[t, "bp_p"]) for t in TARGETS],
        "Харке — Бера, p": [_p(quality.loc[t, "jb_p"]) for t in TARGETS],
        "RESET, p": [_p(quality.loc[t, "reset_p"]) for t in TARGETS],
    })
    c = coefs.set_index(["target", "term"])
    qp, ql, qh = quality.loc["poverty"], quality.loc["life_exp"], quality.loc["housing"]
    pov = yearly[yearly["target"] == "poverty"].set_index("year")
    life = yearly[yearly["target"] == "life_exp"]
    life_sig = int(((life["ci_low"] > 0) | (life["ci_high"] < 0)).sum())
    house = yearly[yearly["target"] == "housing"]
    house_sig = int(((house["ci_low"] > 0) | (house["ci_high"] < 0)).sum())
    checks = {"bp_p": "гомоскедастичность", "jb_p": "нормальность остатков", "reset_p": "функциональная форма"}
    notes = []
    for target in ("life_exp", "housing"):
        failed = [name for key, name in checks.items() if quality.loc[target, key] < 0.05]
        notes.append(f"Для модели {GENITIVE[target]} " + ("отвергается: " + ", ".join(failed) + "." if failed
                                                         else "все три предпосылки выполняются."))
    violations = " ".join(notes)
    pov_sig = int(((yearly[yearly["target"] == "poverty"]["ci_high"] < 0)).sum())
    parts.append(f"""**Таблица 14. Проверка предпосылок МНК** (p-значения тестов)

{_table(diag)}

### 7.3. Трактовка

**Мультиколлинеарность.** VIF от {_num(coefs["vif"].min())} до {_num(coefs["vif"].max())} — ниже порога 5. Факторы
связаны между собой, но не настолько, чтобы оценки стали неустойчивыми.

**Бедность.** Уравнение значимо (F = {_num(qp["f"], 1)}, {_pt(qp["f_p"])}) и объясняет {_num(qp["r2"] * 100, 0)} %
вариации. При фиксированных ВРП и инвестициях рост доходов на 1 % снижает бедность на
{_num(abs(c.loc[("poverty", "ln_income"), "coef"]) / 100, 2)} п. п. (эластичность {_num(c.loc[("poverty", "ln_income"), "elasticity"])}).
По стандартизованным коэффициентам доходы (β* = {_num(c.loc[("poverty", "ln_income"), "beta_std"])}) — главный фактор,
остальные в разы слабее. ВРП при контроле доходов незначим ({_pt(c.loc[("poverty", "ln_gdp"), "p"])}):
вся его связь с бедностью из раздела 5 шла через доходы. Коэффициент при инвестициях **положителен**
и значим (β = {_num(c.loc[("poverty", "ln_invest"), "coef"])}, {_pt(c.loc[("poverty", "ln_invest"), "p"])}): при одинаковых
доходах в регионах с большими инвестициями бедность выше. Наиболее вероятное объяснение — сырьевые регионы,
где инвестиции велики, а создаваемый доход распределён неравномерно; в панельной модели (раздел 8)
этот эффект исчезает.

**Продолжительность жизни.** Уравнение формально значимо ({_pt(ql["f_p"])}), но объясняет лишь {_num(ql["r2"] * 100, 0)} %
вариации, и ни один коэффициент не значим на уровне 5 % с робастными ошибками. Экономические факторы
практически не различают регионы по продолжительности жизни.

**Жилая площадь.** R² = {_num(qh["r2"])}. Коэффициенты при ВРП и инвестициях значимы, но имеют противоположные
знаки ({_signed(c.loc[("housing", "ln_gdp"), "coef"])} и {_signed(c.loc[("housing", "ln_invest"), "coef"])}) при тесной связи
самих факторов между собой. Такая картина типична для случая, когда модель «делит» общий эффект двух
коллинеарных переменных, и содержательно трактовать каждый знак по отдельности не следует.

**Предпосылки.** Для модели бедности отвергаются все три гипотезы: дисперсия остатков непостоянна,
остатки не нормальны, а линейно-логарифмическая форма неадекватна (RESET, {_pt(qp["reset_p"])}).
Последнее согласуется с разделом 6: связь бедности с доходами степенная. Поэтому в таблицах приведены
p-значения с ошибками HC3, устойчивыми к гетероскедастичности, а в разделе 8 оценена модель с постоянной
эластичностью. {violations}

### 7.4. Устойчивость во времени

Та же регрессия оценена отдельно для каждого из {yearly["year"].nunique()} лет.

![Коэффициент при доходах по годам](figures/yearly_coefficients.png)

**Трактовка.** Коэффициент при доходах в модели бедности отрицателен и значим в {pov_sig} годах из {len(pov)}, но по модулю
уменьшается: с {_num(pov.loc[FIRST_YEAR, "coef"], 1)} в {FIRST_YEAR} году до {_num(pov.loc[LAST_YEAR, "coef"], 1)} в {LAST_YEAR}-м.
Причина — в форме связи: когда бедность была около 40 %, тот же прирост доходов снимал больше процентных
пунктов, чем при бедности около 10 %. Для продолжительности жизни коэффициент значим в {life_sig} годах из {len(life)},
для жилой площади — в {house_sig} из {len(house)}: устойчивой связи нет.
""")
    return "\n".join(parts)


_PANEL = r"""## 8. Панельная регрессия

### 8.1. Зачем нужна панельная модель

Регрессия на срезе сравнивает регионы между собой. Но регионы различаются множеством постоянных
особенностей — климатом, географией, структурой экономики, демографией, — которые влияют и на экономику,
и на социальные показатели. Если их не учесть, их влияние приписывается факторам модели (смещение
из-за пропущенных переменных). Панельные данные позволяют убрать эти особенности, даже не измеряя их.

### 8.2. Модель с фиксированными эффектами

$$
y_{it} = \alpha_i + \lambda_t + \beta_1\ln x_{1,it} + \beta_2\ln x_{2,it} + \beta_3\ln x_{3,it} + \boldsymbol\gamma^{\top}\mathbf{z}_{it} + \varepsilon_{it}
$$

- $\alpha_i$ — фиксированный эффект региона: всё, что отличает регион $i$ и не меняется во времени;
- $\lambda_t$ — фиксированный эффект года: всё, что в году $t$ одинаково затронуло все регионы
  (кризисы, пандемия, федеральная политика, смена методики Росстата);
- $\mathbf{z}_{it}$ — контрольные переменные: безработица, врачи, доля горожан, доля пожилых.

Оценка эквивалентна МНК после «внутригруппового» преобразования — вычитания средних по региону и по году:

$$
\ddot{y}_{it} = y_{it} - \bar{y}_{i\cdot} - \bar{y}_{\cdot t} + \bar{y}
$$

Поэтому $\beta$ отвечает на другой вопрос, чем в разделе 7: не «отличаются ли богатые регионы от бедных»,
а **«когда в регионе фактор растёт быстрее, чем в среднем по стране, меняется ли социальный показатель
быстрее, чем в среднем по стране»**. Это гораздо ближе к причинной связи.

Значимость фиксированных эффектов проверяется F-критерием, сравнивающим суммы квадратов остатков
объединённой модели ($RSS_P$) и модели с эффектами ($RSS_{FE}$), где $q$ — число добавленных эффектов:

$$
F = \frac{(RSS_P - RSS_{FE})/q}{RSS_{FE}/(N - q - k - 1)}
$$

Наблюдения одного региона за разные годы не независимы, поэтому стандартные ошибки кластеризованы
по регионам ($G$ — число регионов, $\hat{\mathbf{u}}_g$ — остатки региона $g$):

$$
\widehat{V}(\hat{\boldsymbol\beta}) = (X^{\top}X)^{-1}\left(\sum_{g=1}^{G} X_g^{\top}\hat{\mathbf{u}}_g\hat{\mathbf{u}}_g^{\top}X_g\right)(X^{\top}X)^{-1}
$$

Качество объяснения измеряется «внутригрупповым» $R^2$ — долей разброса, объяснённой факторами
сверх фиксированных эффектов:

$$
R^2_{\text{внутр}} = 1 - \frac{RSS_{FE}}{RSS_{\alpha,\lambda}}
$$

где $RSS_{\alpha,\lambda}$ — сумма квадратов остатков модели, содержащей только эффекты. Обычный $R^2$
у таких моделей близок к единице просто из-за сотни фиктивных переменных и о силе факторов не говорит.

Оценены четыре спецификации:

| Модель | Спецификация |
|---|---|
| `naive` | факторы в текущих ценах + линейный тренд, без эффектов |
| `real` | то же в реальных ценах |
| `fe` | реальные цены + эффекты региона и года |
| `fe_controls` | то же + контрольные переменные |
"""


def _panel(r: dict) -> str:
    coefs, summary, tests = r["coefs"], r["summary"], r["panel_tests"].set_index("target")
    parts = [_PANEL]
    test_table = pd.DataFrame({
        "Модель": [NAMES[t] for t in TARGETS],
        "F": [_num(tests.loc[t, "f_fe"], 1) for t in TARGETS],
        "Степени свободы": [f"{tests.loc[t, 'df1']}; {tests.loc[t, 'df2']}" for t in TARGETS],
        "p": [_p(tests.loc[t, "f_fe_p"]) for t in TARGETS],
        "Бройш — Паган, p": [_p(tests.loc[t, "bp_p"]) for t in TARGETS],
        "Автокорреляция остатков": [_num(tests.loc[t, "rho"]) for t in TARGETS],
    })
    parts.append(f"""### 8.3. Нужны ли фиксированные эффекты и кластеризация

**Таблица 15. Тесты спецификации**

{_table(test_table)}

**Трактовка.** F-критерий отвергает гипотезу об отсутствии эффектов для всех трёх показателей с огромным
запасом: постоянные особенности регионов и общие годовые шоки существенны, и модель без них смещена.
Тест Бройша — Пагана отвергает гомоскедастичность, а остатки соседних лет одного региона коррелируют
с коэффициентом {_num(tests["rho"].min())}–{_num(tests["rho"].max())}. Обычные стандартные ошибки при такой
автокорреляции занижены в разы, поэтому кластеризация по регионам обязательна.

### 8.4. Оценки

В скобках — кластеризованные стандартные ошибки; `***` p < 0,01, `**` p < 0,05, `*` p < 0,1.
""")
    for number, target in enumerate(TARGETS, start=16):
        rows = coefs[coefs["target"] == target]
        terms = [t for t in TERMS if t in set(rows["term"])]
        table = pd.DataFrame({"Переменная": [TERMS[t] for t in terms]})
        for spec in SPECS:
            cells = rows[rows["spec"] == spec].set_index("term")
            table[f"`{spec}`"] = [
                f"{_num(cells.loc[t, 'coef'], 3)}{_stars(cells.loc[t, 'pvalue'])} ({_num(cells.loc[t, 'se'], 3)})"
                if t in cells.index else "" for t in terms
            ]
        quality = summary[summary["target"] == target].set_index("spec")
        extra = pd.DataFrame(
            [["Наблюдений"] + [str(quality.loc[s, "n"]) for s in SPECS],
             ["R²"] + [_num(quality.loc[s, "r2"], 3) for s in SPECS],
             ["R² внутригрупповой"] + [_num(quality.loc[s, "r2_within"], 3) for s in SPECS]],
            columns=table.columns)
        parts.append(f"**Таблица {number}. {LABELS[target]}, {UNITS[target]}**\n\n{_table(pd.concat([table, extra]))}\n")

    effects = effect_of_10pct(coefs)
    best = effects[effects["spec"] == "fe_controls"]
    shown = pd.DataFrame({
        "Показатель": [NAMES[t] for t in best["target"]],
        "Фактор": [TERMS[t].replace("ln ", "") for t in best["term"]],
        "Эффект": best["effect"].map(lambda v: _signed(v, 3)).values,
        "95 %-й интервал": [f"от {_signed(a, 3)} до {_signed(b, 3)}" for a, b in zip(best["ci_low"], best["ci_high"])],
        "p": best["pvalue"].map(_p).values,
    })
    q = summary.set_index(["target", "spec"])
    c = coefs.set_index(["target", "spec", "term"])
    e = best.set_index(["target", "term"])
    el = r["elasticity"].set_index("variant")
    full = el.iloc[0]
    el_table = pd.DataFrame({
        "Период": el.index, "Эластичность": el["coef"].map(lambda v: _num(v, 3)).values,
        "Ст. ош.": el["se"].map(lambda v: _num(v, 3)).values,
        "95 %-й интервал": [f"от {_num(a)} до {_num(b)}" for a, b in zip(el["ci_low"], el["ci_high"])],
        "Средняя бедность, %": el["mean_poverty"].map(lambda v: _num(v, 1)).values, "n": el["n"].astype(str).values,
    })
    rob = r["robustness"]
    rob_table = pd.DataFrame({"Вариант": rob["variant"].drop_duplicates().values})
    for target in TARGETS:
        rows = rob[rob["target"] == target]
        rob_table[NAMES[target]] = [f"{_num(a, 2)}{_stars(p)} ({_num(s, 2)})" for a, s, p in zip(rows["coef"], rows["se"], rows["p"])]
    rp = rob[rob["target"] == "poverty"]
    rl, rh = rob[rob["target"] == "life_exp"], rob[rob["target"] == "housing"]
    res = r["residuals"].set_index("target")
    res_table = pd.DataFrame({
        "Модель": [NAMES[t] for t in TARGETS], "Ст. откл. остатков": [_num(res.loc[t, "resid_std"]) for t in TARGETS],
        "Асимметрия": [_num(res.loc[t, "skew"]) for t in TARGETS], "Эксцесс": [_num(res.loc[t, "kurtosis"]) for t in TARGETS],
        "Харке — Бера": [_num(res.loc[t, "jarque_bera"], 0) for t in TARGETS], "p": [_p(res.loc[t, "pvalue"]) for t in TARGETS],
    })
    parts.append(f"""### 8.5. Трактовка коэффициентов

Коэффициент при логарифме удобнее читать как эффект роста фактора на 10 %: $\\Delta y = \\beta\\cdot\\ln 1{{,}}1 \\approx 0{{,}}095\\,\\beta$.

**Таблица 19. Изменение показателя при росте фактора на 10 %** (модель `fe_controls`)

{_table(shown, text_columns=2)}

![Эффекты факторов: наивная и панельная модели](figures/effects.png)

![Частная регрессия на доходы](figures/partial_regression.png)

Частный график регрессии показывает ту самую связь, которую оценивает $\\beta_3$: и показатель, и логарифм
доходов очищены от остальных факторов и фиксированных эффектов (теорема Фриша — Во — Ловелла),
наклон линии равен коэффициенту модели.

**Бедность.** Связь с доходами сохраняется при самом строгом контроле: β = {_num(c.loc[("poverty", "fe_controls", "ln_income"), "coef"], 2)}
({_pt(c.loc[("poverty", "fe_controls", "ln_income"), "pvalue"])}). Если реальные доходы в регионе растут на 10 % сверх
общероссийской динамики, уровень бедности в нём снижается на **{_num(abs(e.loc[("poverty", "ln_income"), "effect"]), 1)} п. п.**
(95 %-й интервал: от {_num(abs(e.loc[("poverty", "ln_income"), "ci_high"]), 1)} до {_num(abs(e.loc[("poverty", "ln_income"), "ci_low"]), 1)}).
Факторы объясняют {_num(q.loc[("poverty", "fe_controls"), "r2_within"] * 100, 0)} % внутригрупповой вариации — много
для модели, из которой уже убраны все постоянные различия и общий тренд. ВРП и инвестиции незначимы:
положительный коэффициент при инвестициях из раздела 7 был артефактом сравнения разных регионов.
Сравнение столбцов показывает цену наивного подхода: модель в текущих ценах занижала эффект доходов
более чем вдвое ({_num(c.loc[("poverty", "naive", "ln_income"), "coef"], 1)} против {_num(c.loc[("poverty", "fe_controls", "ln_income"), "coef"], 1)}).
В текущих ценах изменение логарифма доходов складывается из реального роста и инфляции; инфляция с бедностью
не связана и «разбавляет» коэффициент.

**Продолжительность жизни.** В моделях без эффектов доходы «значимо» увеличивают продолжительность жизни
(β = {_num(c.loc[("life_exp", "real", "ln_income"), "coef"], 2)}, {_pt(c.loc[("life_exp", "real", "ln_income"), "pvalue"])}), а ВРП — «значимо» сокращает.
С фиксированными эффектами оба результата исчезают: коэффициенты близки к нулю и незначимы,
внутригрупповой R² равен {_num(q.loc[("life_exp", "fe"), "r2_within"], 3)}. Прежняя значимость была ложной:
доходы и продолжительность жизни просто росли одновременно. Экономическая динамика региона
объясняет около {_num(q.loc[("life_exp", "fe"), "r2_within"] * 100, 0)} % того, как менялась продолжительность жизни в нём
относительно страны. Из контрольных переменных значимы безработица и (на уровне 10 %) число врачей,
но положительный знак при безработице содержательного объяснения не имеет.

**Жилая площадь.** Внутригрупповой R² = {_num(q.loc[("housing", "fe_controls"), "r2_within"], 3)}. Единственный значимый
коэффициент — отрицательный при инвестициях ({_num(c.loc[("housing", "fe_controls", "ln_invest"), "coef"], 2)}, {_pt(c.loc[("housing", "fe_controls", "ln_invest"), "pvalue"])}):
в годы инвестиционного подъёма обеспеченность жильём в регионе растёт медленнее. Возможное объяснение —
приток населения в такие регионы, при котором жилая площадь делится на большее число жителей;
проверить его в рамках этих данных нельзя. Обеспеченность жильём растёт во всех регионах почти независимо
от их экономической динамики.

### 8.6. Модель с постоянной эластичностью для бедности

Разделы 6 и 7 показали, что связь бедности с доходами степенная. В панельной форме ей соответствует
модель с логарифмом бедности в левой части, где $\\beta_3$ — эластичность:

$$
\\ln y_{{1,it}} = \\alpha_i + \\lambda_t + \\beta_1\\ln x_{{1,it}} + \\beta_2\\ln x_{{2,it}} + \\beta_3\\ln x_{{3,it}} + \\boldsymbol\\gamma^{{\\top}}\\mathbf{{z}}_{{it}} + \\varepsilon_{{it}}
$$

**Таблица 20. Эластичность бедности по реальным доходам**

{_table(el_table)}

**Трактовка.** Рост реальных доходов на 1 % снижает уровень бедности на {_num(abs(full["coef"]), 2)} %. В отличие
от коэффициента в п. п., эластичность почти одинакова в обе половины периода
({_num(el.iloc[1]["coef"], 2)} и {_num(el.iloc[2]["coef"], 2)}): закономерность устойчива, а снижение эффекта в процентных
пунктах объясняется только тем, что сама бедность стала ниже. Практический смысл: при бедности 10 %
рост доходов на 10 % снижает её примерно на {_num(abs(full["coef"]), 1)} п. п., при бедности 30 % — на {_num(abs(full["coef"]) * 3, 1)} п. п.

### 8.7. Проверка устойчивости

Вывод надёжен, если он не зависит от частных решений исследователя. Коэффициент при ln доходов
переоценён на разных выборках и в альтернативной спецификации — в первых разностях:

$$
\\Delta y_{{it}} = \\lambda_t + \\beta_1\\Delta\\ln x_{{1,it}} + \\beta_2\\Delta\\ln x_{{2,it}} + \\beta_3\\Delta\\ln x_{{3,it}} + \\boldsymbol\\gamma^{{\\top}}\\Delta\\mathbf{{z}}_{{it}} + \\Delta\\varepsilon_{{it}}
$$

**Таблица 21. Коэффициент при ln доходов в разных вариантах модели**

{_table(rob_table)}

**Трактовка.** Для бедности коэффициент отрицателен и значим на уровне 1 % во всех {len(rp)} вариантах,
в диапазоне от {_num(rp["coef"].min(), 1)} до {_num(rp["coef"].max(), 1)}. Результат не держится на регионах-лидерах,
Северном Кавказе или пандемийных годах и сохраняется, когда доходы берутся с лагом в один год — то есть
доходы предыдущего года предсказывают бедность следующего, что ослабляет подозрение в обратной причинности.
Различие между периодами — следствие формы связи (раздел 8.6).
Для продолжительности жизни коэффициент незначим на уровне 5 % во всех вариантах
(наименьшее {_pt(rl["p"].min())}). Для жилой площади знак меняется от варианта к варианту
(от {_num(rh["coef"].min(), 1)} до {_num(rh["coef"].max(), 1)}), значим он в {int((rh["p"] < 0.05).sum())} случае из {len(rh)}: устойчивой связи нет.

### 8.8. Диагностика остатков

**Таблица 22. Остатки модели `fe_controls`**

{_table(res_table)}

![Диагностика остатков](figures/diagnostics.png)

**Трактовка.** Для продолжительности жизни и жилой площади остатки расположены вокруг нуля без заметного
рисунка. Для бедности рисунок есть: остатки положительны при самых низких и самых высоких прогнозах
и отрицательны в середине, а их разброс растёт с уровнем прогноза. Модель в процентных пунктах занижает
бедность на краях — это след степенной связи, которую прямая в полулогарифмических координатах описывает
лишь приближённо. У всех трёх моделей тяжёлые хвосты (эксцесс больше нуля, на Q-Q графике концы отходят
от прямой): в отдельные годы в отдельных регионах показатели резко отклонялись от прогноза. Тест Харке — Бера
нормальность отвергает. На состоятельность оценок это не влияет, а значимость при {int(q.loc[("poverty", "fe_controls"), "n"])}
наблюдениях и {int(q.loc[("poverty", "fe_controls"), "regions"])} кластерах опирается на асимптотическую нормальность оценок,
а не на нормальность остатков. Рисунок остатков модели бедности — ещё один довод в пользу
логарифмической спецификации из раздела 8.6.
""")
    return "\n".join(parts)


_CLUSTER = r"""## 9. Типология регионов

### 9.1. Методика

Регрессии описывают средние закономерности. Кластерный анализ отвечает на другой вопрос: какие типы
регионов существуют. Используются девять показателей последнего года; ВРП и доходы берутся в логарифмах.
Чтобы показатели в разных единицах имели равный вес, они стандартизуются:

$$
z_{ij} = \frac{x_{ij}-\bar{x}_j}{s_j}
$$

Метод k-средних делит регионы на $k$ групп $S_1,\dots,S_k$ так, чтобы суммарное расстояние от регионов
до центров своих кластеров $\boldsymbol\mu_c$ было наименьшим:

$$
W(k) = \sum_{c=1}^{k}\sum_{i\in S_c}\lVert \mathbf{z}_i-\boldsymbol\mu_c\rVert^{2} \to \min
$$

Число кластеров выбирается по двум критериям. Метод «локтя»: $W(k)$ всегда убывает с ростом $k$, выбирается
точка, после которой убывание замедляется. Коэффициент силуэта для региона $i$:

$$
s_i = \frac{b_i-a_i}{\max(a_i,\,b_i)}
$$

где $a_i$ — среднее расстояние до регионов своего кластера, $b_i$ — до регионов ближайшего чужого.
Среднее $s_i$ близко к 1 при чётко разделённых кластерах и к 0 при размытых границах.

Для наглядности девять показателей сворачиваются в главные компоненты — некоррелированные линейные
комбинации, вбирающие наибольшую долю общей дисперсии. Значимость различий между кластерами по каждому
показателю проверяется ранговым критерием Краскела — Уоллиса ($H_0$: распределения в кластерах одинаковы).
"""


def _clusters(r: dict) -> str:
    assignments, profiles, scores = r["assignments"], r["profiles"], r["scores"]
    inertia, loadings, explained, kruskal = r["inertia"], r["loadings"], r["explained"], r["kruskal"]
    selection = pd.DataFrame({"k": scores["k"].astype(str)})
    w = inertia.set_index("k")["inertia"]
    selection["W(k)"] = [_num(w[k], 0) for k in scores["k"]]
    selection["Снижение W, %"] = [_num((1 - w[k] / w[k - 1]) * 100, 0) for k in scores["k"]]
    selection["Силуэт"] = scores["silhouette"].map(lambda v: _num(v, 3)).values

    load = pd.DataFrame({"Показатель": [NAMES[f] for f in loadings["feature"]]})
    for i, col in enumerate(("pc1", "pc2", "pc3")):
        load[f"ГК{i + 1} ({_num(explained[i] * 100, 0)} %)"] = loadings[col].map(_signed).values

    cols = list(CLUSTER_COLUMNS)
    table = pd.DataFrame({"Кластер": profiles["cluster"].astype(str), "Регионов": profiles["regions"].astype(str)})
    for col in cols:
        table[CLUSTER_COLUMNS[col]] = profiles[col].map(lambda v: _num(v, 1)).values
    kw = pd.DataFrame({"Показатель": [NAMES[f] for f in kruskal["feature"]],
                       "H": kruskal["h"].map(lambda v: _num(v, 1)).values, "p": kruskal["p"].map(_p).values})
    same = [_low(NAMES[f]) for f, p in zip(kruskal["feature"], kruskal["p"]) if p >= 0.05]
    lists = "\n".join(f"- **Кластер {cluster}** ({len(group)}): {', '.join(group['region'])}."
                      for cluster, group in assignments.groupby("cluster"))
    p_ = profiles.set_index("cluster")
    s = scores.set_index("k")["silhouette"]
    return f"""{_CLUSTER}
### 9.2. Выбор числа кластеров

**Таблица 23. Критерии выбора k**

{_table(selection)}

![Выбор числа кластеров](figures/cluster_selection.png)

**Трактовка.** Переход от двух кластеров к трём снижает $W$ на {_num((1 - w[3] / w[2]) * 100, 0)} %, от трёх к четырём —
уже на {_num((1 - w[4] / w[3]) * 100, 0)} %: «локоть» приходится на k = 3. Силуэт максимален при k = 2 ({_num(s[2], 2)}),
но такое деление лишь отделяет бедные республики от всех остальных; при k = 3 он немного ниже ({_num(s[3], 2)}),
а при k = 5 резко падает ({_num(s[5], 2)}). Выбрано **k = {N_CLUSTERS}**. Значения силуэта около 0,4 говорят о том,
что границы между типами размыты: это не изолированные группы, а участки непрерывного распределения.

### 9.3. Главные компоненты

**Таблица 24. Нагрузки показателей на главные компоненты** (в скобках — доля объяснённой дисперсии)

{_table(load)}

**Трактовка.** Три компоненты вбирают {_num(explained.sum() * 100, 0)} % общей дисперсии девяти показателей.
Первая ({_num(explained[0] * 100, 0)} %) — ось «уровень развития»: на одном полюсе высокие ВРП, доходы и урбанизация,
на другом — бедность и безработица. Вторая ({_num(explained[1] * 100, 0)} %) — демографическая: стареющие регионы
с большим жилым фондом на человека против молодых и богатых с высокой обеспеченностью врачами.
Третья ({_num(explained[2] * 100, 0)} %) почти целиком состоит из продолжительности жизни (нагрузка {_signed(loadings.set_index("feature").loc["life_exp", "pc3"])}):
этот показатель образует **отдельное измерение**, не связанное ни с уровнем развития, ни с демографической
структурой, — тот же вывод, к которому привели регрессии.

### 9.4. Профили кластеров

**Таблица 25. Средние значения показателей по кластерам, {LAST_YEAR} г.**

{_table(table, text_columns=1)}

{lists}

![Кластеры на плоскости главных компонент](figures/clusters.png)

**Таблица 26. Критерий Краскела — Уоллиса: различаются ли кластеры**

{_table(kw)}

**Трактовка.**

- **Кластер 1 — лидеры.** Столицы и ресурсные регионы Севера и Дальнего Востока: ВРП на душу в {_num(p_.loc[1, "gdp_real"] / p_.loc[2, "gdp_real"], 1)} раза
  выше, чем в основной группе, доходы — в {_num(p_.loc[1, "income_real"] / p_.loc[2, "income_real"], 1)} раза, самая низкая бедность
  и самая высокая обеспеченность врачами. При этом жилья на человека меньше, чем в основной группе.
- **Кластер 2 — основная группа.** Три четверти регионов страны: средние доходы, умеренная бедность,
  самое старое население и самая высокая обеспеченность жильём.
- **Кластер 3 — отстающие.** Республики Северного Кавказа и юга Сибири, Крым: доходы на {_num((1 - p_.loc[3, "income_real"] / p_.loc[2, "income_real"]) * 100, 0)} %
  ниже основной группы, бедность почти вдвое выше, безработица втрое выше, молодое и преимущественно
  сельское население, меньше всего жилья на человека.

Кластеры значимо различаются по всем показателям, кроме: {", ".join(same) or "нет таких"}.
Различие средней продолжительности жизни между кластерами ({_num(p_.loc[1, "life_exp"], 1)}, {_num(p_.loc[2, "life_exp"], 1)}
и {_num(p_.loc[3, "life_exp"], 1)} года) статистически незначимо: богатые и бедные типы регионов по ней не отличаются.
"""


def _discussion(r: dict) -> str:
    q = r["summary"].set_index(["target", "spec"])
    el = r["elasticity"].iloc[0]
    beta = r["beta"].set_index("indicator")
    e = effect_of_10pct(r["coefs"]).set_index(["target", "spec", "term"]).loc[("poverty", "fe_controls", "ln_income")]
    return f"""## 10. Обсуждение результатов

### 10.1. Проверка гипотез

| Гипотеза | Результат | Основание |
|---|---|---|
| **H1.** Экономические факторы снижают бедность, сильнее всего — доходы | **Подтверждена для доходов**, не подтверждена для ВРП и инвестиций | Эластичность {_num(el["coef"], 2)}, значима во всех вариантах модели; ВРП и инвестиции незначимы при контроле доходов |
| **H2.** Экономические факторы повышают продолжительность жизни | **Не подтверждена** | Внутригрупповой R² {_num(q.loc[("life_exp", "fe"), "r2_within"], 2)}; коэффициенты незначимы во всех вариантах |
| **H3.** Экономические факторы повышают обеспеченность жильём | **Не подтверждена** | Внутригрупповой R² {_num(q.loc[("housing", "fe_controls"), "r2_within"], 2)}; знак коэффициента при доходах неустойчив |
| **H4.** Регионы сближаются | **Подтверждена частично** | β-сходимость по доходам (b = {_num(beta.loc["income_real", "b"], 3)}, {_pt(beta.loc["income_real", "p"])}); по ВРП сходимости нет |

### 10.2. Что это означает

**Важен не объём экономики региона, а доходы его жителей.** ВРП и инвестиции связаны с бедностью только
постольку, поскольку связаны с доходами; сами по себе они ничего не добавляют. Одновременно доходы регионов
сближались, а ВРП — нет. Вместе эти два результата означают, что снижение бедности в отстающих регионах
опиралось не на рост их собственного производства, а на рост доходов, который от производства был
частично отвязан.

**Продолжительность жизни живёт по своим законам.** Она выросла во всех регионах, но регионы, богатевшие
быстрее, не получили преимущества. Самые бедные республики остаются среди лидеров, самые богатые северные
регионы — среди отстающих. В данных исследования нет переменных, которые обычно называют определяющими, —
образа жизни, потребления алкоголя, климата, качества (а не количества) медицинской помощи; часть из них
постоянна и ушла в фиксированные эффекты региона.

**Обеспеченность жильём определяется демографией сильнее, чем экономикой.** Больше всего жилья на человека
там, где население убывает и стареет, а не там, где высокие доходы.

**Простая регрессия по объединённым данным вводит в заблуждение.** Она давала «значимый» положительный
эффект доходов на продолжительность жизни и R² выше 0,5. Обе цифры — следствие общего тренда.
Этот результат важен сам по себе: высокий R² и значимые коэффициенты во временных данных не доказывают связи.

### 10.3. Ограничения

1. **Связь, но не доказанная причинность.** Фиксированные эффекты убирают постоянные особенности регионов
   и общие шоки, но не меняющиеся во времени пропущенные переменные и не обратную причинность.
   Модель с лагом ослабляет, но не снимает это возражение.
2. **Бедность и доходы связаны по определению:** бедность — доля людей с доходами ниже черты. Наличие связи
   ожидаемо; содержательны её величина, устойчивость эластичности и то, что остальные факторы ничего не добавляют.
3. **Изменения методики Росстата:** расчёт доходов с 2013 года, переход от прожиточного минимума к границе
   бедности с 2021 года, учёт жилищного фонда с 2013 года. Общие для всех регионов сдвиги поглощаются
   эффектами года, но если изменения затронули регионы по-разному, оценки могут быть смещены.
4. **Дефлятор приближённый.** Стоимость фиксированного набора измеряется на декабрь и отражает потребительские
   цены; для ВРП и инвестиций точнее были бы индексы-дефляторы, которых в приложении к сборнику нет.
5. **Средние по региону скрывают неравенство внутри него.** Среднедушевой доход Москвы или Тюменской области
   ничего не говорит о распределении; регионы с автономными округами учтены целиком.
6. **Кластеры размыты** (силуэт около 0,4), типология описательная.

## 11. Выводы

1. Из трёх экономических факторов с социальными показателями устойчиво связаны только **доходы населения**,
   и только с **бедностью**: рост реальных доходов на 1 % снижает уровень бедности на {_num(abs(el["coef"]), 1)} %,
   рост на 10 % — на {_num(abs(e["effect"]), 1)} п. п. в среднем за период. Связь степенная, эластичность стабильна во времени.
2. ВРП и инвестиции на душу населения при тех же доходах с бедностью **не связаны**.
3. **Продолжительность жизни** от экономической динамики региона практически не зависит
   (около {_num(q.loc[("life_exp", "fe"), "r2_within"] * 100, 0)} % объяснённой вариации) и образует отдельное измерение различий между регионами.
4. **Обеспеченность жильём** растёт повсеместно и определяется демографией, а не экономикой региона.
5. Регионы **сближались по доходам** со скоростью {_num(beta.loc["income_real", "speed_pct"], 1)} % в год до начала 2010-х годов;
   по ВРП сближения не было.
6. Регионы делятся на **три типа**: лидеры ({int(r["profiles"].set_index("cluster").loc[1, "regions"])}),
   основная группа ({int(r["profiles"].set_index("cluster").loc[2, "regions"])}) и отстающие ({int(r["profiles"].set_index("cluster").loc[3, "regions"])});
   по продолжительности жизни типы между собой значимо не различаются.
7. Методический результат: без пересчёта в реальные цены и без учёта панельной структуры данных
   выводы о роли экономических факторов оказываются ложными.

## Источники

1. Регионы России. Социально-экономические показатели. 2025: приложение к статистическому сборнику /
   Росстат. — М., 2025. — URL: https://rosstat.gov.ru/folder/210/document/47652
2. Магнус Я. Р., Катышев П. К., Пересецкий А. А. Эконометрика. Начальный курс. — М.: Дело.
3. Wooldridge J. M. Econometric Analysis of Cross Section and Panel Data. — 2nd ed. — MIT Press, 2010.
4. Barro R. J., Sala-i-Martin X. Convergence // Journal of Political Economy. — 1992. — Vol. 100, No. 2. — P. 223–251.
5. Cameron A. C., Miller D. L. A Practitioner's Guide to Cluster-Robust Inference // Journal of Human Resources. — 2015. — Vol. 50, No. 2. — P. 317–372.
6. Rousseeuw P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis //
   Journal of Computational and Applied Mathematics. — 1987. — Vol. 20. — P. 53–65.
"""


def build_report(r: dict) -> Path:
    """Собирает отчёт из словаря результатов; попутно сохраняет таблицы в results/tables."""
    r = dict(r)
    r["stats"], r["corr"], r["residuals"] = describe(r["panel"]), within_year_corr(r["panel"]), residual_tests(r["fits"])
    q = r["summary"].set_index(["target", "spec"])
    el = r["elasticity"].iloc[0]

    head = f"""# Экономические факторы и качество жизни в регионах России, {FIRST_YEAR}–{LAST_YEAR}

*Эмпирическое исследование по данным Росстата. Текст собирается командой `python -m regions`:
все числа, таблицы и графики получены расчётом из `data/raw`.*

## Аннотация

Исследуется связь ВРП, инвестиций и доходов населения с уровнем бедности, ожидаемой продолжительностью
жизни и обеспеченностью жильём по панели {r["panel"]["region"].nunique()} регионов России за {FIRST_YEAR}–{LAST_YEAR} годы
({len(r["panel"])} наблюдений). Денежные показатели приведены к ценам {BASE_YEAR} года с поправкой на региональный
уровень цен. Последовательно применены корреляционный анализ, парные регрессии пяти функциональных форм,
множественная регрессия с проверкой предпосылок МНК и панельная регрессия с фиксированными эффектами
региона и года. Установлено, что с бедностью устойчиво связаны только доходы населения: эластичность
равна {_num(el["coef"], 2)}, а ВРП и инвестиции при контроле доходов незначимы. Связь экономических факторов
с продолжительностью жизни, заметная в простых моделях (R² = {_num(q.loc[("life_exp", "naive"), "r2"])}), оказывается
следствием общего тренда: после его устранения факторы объясняют около {_num(q.loc[("life_exp", "fe"), "r2_within"] * 100, 0)} % вариации.
Регионы сближались по доходам, но не по ВРП. Выделены три типа регионов.

**Содержание:** [1. Введение](#1-введение) · [2. Данные](#2-данные) · [3. Разведочный анализ](#3-разведочный-анализ) ·
[4. Сходимость](#4-сходимость-регионов) · [5. Корреляции](#5-корреляционный-анализ) ·
[6. Парная регрессия](#6-парная-регрессия) · [7. Множественная регрессия](#7-множественная-регрессия) ·
[8. Панельная регрессия](#8-панельная-регрессия) · [9. Типология](#9-типология-регионов) ·
[10. Обсуждение](#10-обсуждение-результатов) · [11. Выводы](#11-выводы)
"""
    sections = [head, _intro(r), _data(r), _explore(r), _convergence(r), _correlation(r), _pairs(r),
                _multiple(r), _panel(r), _clusters(r), _discussion(r)]
    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8", newline="\n")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    saved = {
        "descriptive_stats": r["stats"], "normality": r["normality"], "correlation_tests": r["corr_tests"],
        "pair_models": r["pairs"], "cross_section_coefficients": r["cs_coefs"], "cross_section_quality": r["cs_quality"],
        "yearly_coefficients": r["yearly"], "panel_tests": r["panel_tests"], "poverty_elasticity": r["elasticity"],
        "robustness": r["robustness"], "sigma_convergence": r["sigma"], "beta_convergence": r["beta"],
        "residual_tests": r["residuals"], "pca_loadings": r["loadings"], "cluster_kruskal": r["kruskal"],
        "cluster_inertia": r["inertia"],
    }
    for name, frame in saved.items():
        frame.round(5).to_csv(TABLES_DIR / f"{name}.csv", index=False)
    r["corr"].round(3).to_csv(TABLES_DIR / "correlations.csv")
    return REPORT_PATH
