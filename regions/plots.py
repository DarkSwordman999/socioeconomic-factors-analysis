"""Графики для отчёта. Все рисунки сохраняются в results/figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from regions.indicators import LABELS, TARGETS, UNITS
from regions.models import SPECS, effect_of_10pct

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

# Палитра: фиксированный порядок категориальных цветов, нейтральные оси и сетка
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "#cde2fb"
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#eb6834", "#f0efec", "#2a78d6"])

FACTOR_LABELS = {"ln_gdp": "ВРП на душу", "ln_invest": "Инвестиции на душу", "ln_income": "Доходы на душу"}
SHORT = {
    "gdp_real": "ВРП", "invest_real": "Инвестиции", "income_real": "Доходы", "poverty": "Бедность",
    "life_exp": "Продолж. жизни", "housing": "Жильё", "infant_mort": "Младенч. смертность",
    "unemployment": "Безработица", "doctors": "Врачи", "urban": "Горожане", "elderly": "Пожилые",
    "students": "Студенты",
}

SHORT_TARGETS = {"poverty": "Уровень бедности", "life_exp": "Продолжительность жизни",
                 "housing": "Жилая площадь на жителя"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _label_points(ax, rows: pd.DataFrame, x: str, y: str, x_scale: float = 1.0) -> None:
    """Подписывает точки по порядку, пропуская те, чья подпись легла бы на уже поставленную."""
    ax.figure.canvas.draw()
    placed = []
    for _, row in rows.iterrows():
        px, py = ax.transData.transform((row[x] * x_scale, row[y]))
        if any(abs(py - qy) < 16 and abs(px - qx) < 260 for qx, qy in placed):
            continue
        placed.append((px, py))
        ax.annotate(row["region"], (row[x] * x_scale, row[y]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK_2)


def plot_trends(panel: pd.DataFrame, russia: pd.DataFrame) -> Path:
    """Динамика социальных показателей: разброс регионов и Россия в целом."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key in zip(axes, TARGETS):
        stats = panel.groupby("year")[key].quantile([0.1, 0.5, 0.9]).unstack()
        ax.fill_between(stats.index, stats[0.1], stats[0.9], color=BAND, linewidth=0,
                        label="80 % регионов")
        ax.plot(stats.index, stats[0.5], color=SERIES[0], label="Медианный регион")
        ax.plot(russia["year"], russia[key], color=SERIES[1], label="Россия в целом")
        ax.set_title(SHORT_TARGETS[key])
        ax.set_ylabel(UNITS[key])
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    return _save(fig, "trends.png")


def plot_nominal_vs_real(russia: pd.DataFrame) -> Path:
    """Номинальные и реальные доходы России, 2000 = 100."""
    base = russia.set_index("year")
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label, color in [("income", "В текущих ценах", SERIES[1]),
                              ("income_real", "В ценах 2023 г.", SERIES[0])]:
        index = base[col] / base[col].iloc[0] * 100
        ax.plot(index.index, index, color=color, label=label)
        ax.annotate(f"×{index.iloc[-1] / 100:.1f}", (index.index[-1], index.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK_2)
    ax.set_title("Среднедушевые доходы в России, 2000 = 100")
    ax.set_yscale("log")
    ax.set_yticks([100, 200, 500, 1000, 2000], labels=["100", "200", "500", "1000", "2000"])
    ax.minorticks_off()
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    return _save(fig, "nominal_vs_real.png")


def plot_income_poverty(panel: pd.DataFrame, years=(2005, 2023)) -> Path:
    """Связь реальных доходов и бедности в начале и в конце периода."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for year, color in zip(years, (SERIES[1], SERIES[0])):
        data = panel[panel["year"] == year].dropna(subset=["income_real", "poverty"])
        ax.scatter(data["income_real"] / 1000, data["poverty"], s=34, color=color,
                   edgecolor=SURFACE, linewidth=1, label=str(year))
    ax.set_xscale("log")
    ax.set_xlim(10, 160)
    ax.set_xticks([10, 20, 30, 50, 80, 120], labels=["10", "20", "30", "50", "80", "120"])
    ax.minorticks_off()
    last = panel[panel["year"] == years[-1]].dropna(subset=["income_real", "poverty"])
    _label_points(ax, pd.concat([last.nlargest(3, "poverty"), last.nlargest(3, "income_real")]),
                  "income_real", "poverty", x_scale=1 / 1000)
    ax.set_xlabel("Доходы на душу, тыс. руб. в месяц в ценах 2023 г. (лог. шкала)")
    ax.set_ylabel("Уровень бедности, %")
    ax.set_title("Чем выше реальные доходы, тем ниже бедность")
    ax.legend(title="Год", loc="upper right")
    return _save(fig, "income_poverty.png")


EFFECT_SPECS = {
    "naive": "Наивная модель: текущие цены, все наблюдения в одной выборке",
    "fe_controls": "Панельная модель: реальные цены, эффекты региона и года, контрольные переменные",
}


def plot_effects(coefs: pd.DataFrame, specs=("naive", "fe_controls")) -> Path:
    """Эффект роста фактора на 10 %: наивная модель против панельной."""
    effects = effect_of_10pct(coefs)
    terms = list(FACTOR_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, target in zip(axes, TARGETS):
        for shift, (spec, color) in zip((0.16, -0.16), zip(specs, (SERIES[1], SERIES[0]))):
            rows = effects[(effects["target"] == target) & (effects["spec"] == spec)].set_index("term").loc[terms]
            ys = np.arange(len(terms)) + shift
            ax.hlines(ys, rows["ci_low"], rows["ci_high"], color=color, linewidth=2)
            ax.scatter(rows["effect"], ys, s=46, color=color, edgecolor=SURFACE, linewidth=1.2,
                       zorder=3, label=EFFECT_SPECS.get(spec, SPECS[spec]["title"]))
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(range(len(terms)), [FACTOR_LABELS[t] for t in terms])
        ax.tick_params(axis="y", colors=INK_2)
        ax.set_title(SHORT_TARGETS[target])
        ax.set_xlabel(UNITS[target])
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.26), ncol=1)
    fig.suptitle("Изменение показателя при росте фактора на 10 % (с 95 %-м доверительным интервалом)",
                 x=0.125, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.82)
    return _save(fig, "effects.png")


def plot_clusters(assignments: pd.DataFrame, year: int) -> Path:
    """Регионы на плоскости двух главных компонент, цвет — кластер."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster, group in assignments.groupby("cluster"):
        ax.scatter(group["pc1"], group["pc2"], s=44, color=SERIES[cluster - 1], edgecolor=SURFACE,
                   linewidth=1, label=f"Кластер {cluster} ({len(group)})")
    centre = assignments[["pc1", "pc2"]].mean()
    distance = np.hypot(assignments["pc1"] - centre["pc1"], assignments["pc2"] - centre["pc2"])
    ax.set_xlim(assignments["pc1"].min() - 0.5, assignments["pc1"].max() + 2.6)
    _label_points(ax, assignments.loc[distance.nlargest(22).index], "pc1", "pc2")
    ax.set_xlabel("Первая главная компонента")
    ax.set_ylabel("Вторая главная компонента")
    ax.set_title(f"Типология регионов по социально-экономическому профилю, {year} г.")
    ax.legend(loc="best")
    return _save(fig, "clusters.png")


def plot_convergence(panel: pd.DataFrame) -> Path:
    """Неравенство регионов по доходам: коэффициент вариации по годам."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label, color in [("income", "Доходы в текущих ценах", SERIES[1]),
                              ("income_real", "Доходы в реальных ценах", SERIES[0]),
                              ("gdp_real", "ВРП на душу в реальных ценах", SERIES[2])]:
        by_year = panel.groupby("year")[col]
        variation = by_year.std() / by_year.mean() * 100
        ax.plot(variation.index, variation, color=color, label=label)
    ax.set_ylim(0, None)
    ax.set_ylabel("Коэффициент вариации, %")
    ax.set_title("Разрыв между регионами: доходы сблизились, ВРП — нет")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    return _save(fig, "convergence.png")


def plot_correlations(panel: pd.DataFrame) -> Path:
    """Корреляции показателей внутри года (общий тренд вычтен)."""
    cols = ["gdp_real", "invest_real", "income_real", "poverty", "life_exp", "housing",
            "infant_mort", "unemployment", "doctors", "urban", "elderly", "students"]
    data = panel[["year"] + cols].copy()
    data[["gdp_real", "invest_real", "income_real"]] = np.log(data[["gdp_real", "invest_real", "income_real"]])
    corr = (data[cols] - data.groupby("year")[cols].transform("mean")).corr()

    fig, ax = plt.subplots(figsize=(8.4, 7))
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    labels = [SHORT[c] for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=45, ha="right", color=INK_2)
    ax.set_yticks(range(len(cols)), labels, color=INK_2)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}".replace("0.", ".").replace("1.00", "1"),
                    ha="center", va="center", fontsize=7.5,
                    color=SURFACE if abs(corr.iat[i, j]) > 0.75 else INK)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Корреляции показателей между регионами (внутри года)")
    fig.colorbar(image, ax=ax, shrink=0.7, pad=0.02).outline.set_visible(False)
    return _save(fig, "correlations.png")


MAIN = ["gdp_real", "invest_real", "income_real", "poverty", "life_exp", "housing"]


def plot_distributions(panel: pd.DataFrame) -> Path:
    """Гистограммы основных показателей по всем наблюдениям."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, key in zip(axes.flat, MAIN):
        values = panel[key].dropna()
        money = key.endswith("_real")
        if money:
            values = values / 1000
            bins = np.geomspace(values.min(), values.max(), 36)
            ax.set_xscale("log")
            ticks = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if values.min() <= t <= values.max()]
            ax.set_xticks(ticks, labels=[str(t) for t in ticks])
            ax.minorticks_off()
        else:
            bins = 36
        ax.hist(values, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(values.median(), color=SERIES[1], linewidth=2)
        ax.set_title(SHORT[key] if key not in SHORT_TARGETS else SHORT_TARGETS[key])
        ax.set_xlabel(("тыс. " + UNITS[key] + ", лог. шкала") if money else UNITS[key])
        ax.grid(axis="x", visible=False)
    axes[0, 0].plot([], [], color=SERIES[1], label="Медиана")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Распределения показателей, все регионы и годы", x=0.125, ha="left",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(hspace=0.5, top=0.9)
    return _save(fig, "distributions.png")


def plot_boxplots(panel: pd.DataFrame, year: int) -> Path:
    """Диаграммы размаха социальных показателей по федеральным округам."""
    data = panel[panel["year"] == year]
    order = data.groupby("district")["income_real"].median().sort_values().index
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, key in zip(axes, TARGETS):
        groups = [data.loc[data["district"] == d, key].dropna() for d in order]
        ax.boxplot(groups, vert=False, widths=0.55, patch_artist=True,
                   boxprops=dict(facecolor=BAND, edgecolor=SERIES[0]),
                   medianprops=dict(color=SERIES[0], linewidth=2),
                   whiskerprops=dict(color=SERIES[0]), capprops=dict(color=SERIES[0]),
                   flierprops=dict(marker="o", markersize=4, markerfacecolor=SERIES[1], markeredgecolor=SURFACE))
        ax.set_yticks(range(1, len(order) + 1), list(order))
        ax.tick_params(axis="y", colors=INK_2)
        ax.set_title(SHORT_TARGETS[key])
        ax.set_xlabel(UNITS[key])
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Разброс регионов внутри федеральных округов, {year} г.", x=0.125, ha="left",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.84)
    return _save(fig, "boxplots.png")


def plot_diagnostics(fits: dict) -> Path:
    """Панельная модель: факт и прогноз, остатки против прогноза, Q-Q график остатков."""
    from scipy import stats

    fig, axes = plt.subplots(3, 3, figsize=(13, 11.5))
    for col, (target, model) in enumerate(fits.items()):
        actual, fitted, resid = model.model.endog, model.fittedvalues, model.resid
        ax = axes[0, col]
        ax.scatter(fitted, actual, s=10, color=SERIES[0], alpha=0.45, linewidth=0)
        low, high = min(actual.min(), fitted.min()), max(actual.max(), fitted.max())
        ax.plot([low, high], [low, high], color=SERIES[1], linewidth=1.5)
        ax.set_title(SHORT_TARGETS[target] + ": факт и прогноз", fontsize=10)
        ax.set_xlabel(f"Прогноз модели, {UNITS[target]}")
        ax.set_ylabel(f"Факт, {UNITS[target]}")

        ax = axes[1, col]
        ax.scatter(fitted, resid, s=10, color=SERIES[0], alpha=0.45, linewidth=0)
        ax.axhline(0, color=SERIES[1], linewidth=1.5)
        ax.set_title("Остатки против прогноза", fontsize=10)
        ax.set_xlabel(f"Прогноз модели, {UNITS[target]}")
        ax.set_ylabel(f"Остатки, {UNITS[target]}")

        ax = axes[2, col]
        (theory, ordered), (slope, intercept, _) = stats.probplot(resid, dist="norm")
        ax.scatter(theory, ordered, s=10, color=SERIES[0], alpha=0.45, linewidth=0)
        ax.plot(theory, slope * theory + intercept, color=SERIES[1], linewidth=1.5)
        ax.set_title("Q-Q график остатков", fontsize=10)
        ax.set_xlabel("Квантили нормального распределения")
        ax.set_ylabel(f"Остатки, {UNITS[target]}")
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    return _save(fig, "diagnostics.png")


FORM_NAMES = {"linear": "линейная", "log": "логарифмическая", "power": "степенная",
              "exp": "экспоненциальная", "quadratic": "квадратичная"}
FACTOR_SHORT = {"gdp_real": "ВРП на душу", "invest_real": "Инвестиции на душу", "income_real": "Доходы на душу"}


def plot_pair_regressions(panel: pd.DataFrame, year: int, pairs: pd.DataFrame, curves: dict) -> Path:
    """Поля корреляции 3×3 с линиями парной регрессии: линейной и лучшей по скорр. R²."""
    data = panel[panel["year"] == year]
    factors = list(FACTOR_SHORT)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for i, target in enumerate(TARGETS):
        for j, factor in enumerate(factors):
            ax = axes[i, j]
            pair = data[[factor, target]].dropna()
            x = pair[factor] / 1000
            ax.scatter(x, pair[target], s=22, color=SERIES[0], alpha=0.7, edgecolor=SURFACE, linewidth=0.6)
            grid = np.geomspace(x.min(), x.max(), 200)
            rows = pairs[(pairs["target"] == target) & (pairs["factor"] == factor)].set_index("form")
            best = rows.index[rows["best"]][0]
            ax.plot(grid, curves[(target, factor, "linear")](grid), color=MUTED, linewidth=1.5,
                    linestyle="--", label="Линейная")
            ax.plot(grid, curves[(target, factor, best)](grid), color=SERIES[1], linewidth=2,
                    label="Лучшая форма")
            ax.set_xscale("log")
            ticks = [t for t in (20, 30, 40, 50, 70, 100, 200, 500, 1000, 2000, 5000) if x.min() <= t <= x.max()]
            ax.set_xticks(ticks, labels=[str(t) for t in ticks])
            ax.minorticks_off()
            ax.set_ylim(pair[target].min() - 0.08 * np.ptp(pair[target]), pair[target].max() + 0.22 * np.ptp(pair[target]))
            ax.text(0.03, 0.96, f"{FORM_NAMES[best]}: R² = {rows.loc[best, 'r2']:.2f}\nлинейная: R² = {rows.loc['linear', 'r2']:.2f}",
                    transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2)
            if i == 0:
                ax.set_title(FACTOR_SHORT[factor])
            if i == 2:
                ax.set_xlabel("тыс. руб. в ценах 2023 г. (лог. шкала)")
            if j == 0:
                ax.set_ylabel(f"{SHORT_TARGETS[target]}, {UNITS[target]}")
    axes[0, 2].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(f"Парные регрессии: социальный показатель от экономического фактора, {year} г.",
                 x=0.125, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.93, hspace=0.22, wspace=0.24)
    return _save(fig, "pair_regressions.png")


def plot_partial(partials: dict, coefs: pd.DataFrame) -> Path:
    """Частные графики регрессии: связь с доходами после вычета остальных регрессоров и эффектов."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, target in zip(axes, TARGETS):
        points = partials[target]
        row = coefs[(coefs["target"] == target) & (coefs["spec"] == "fe_controls") & (coefs["term"] == "ln_income")].iloc[0]
        ax.scatter(points["x"], points["y"], s=9, color=SERIES[0], alpha=0.4, linewidth=0)
        grid = np.array([points["x"].min(), points["x"].max()])
        ax.plot(grid, row["coef"] * grid, color=SERIES[1], linewidth=2)
        p_text = "p < 0,001" if row["pvalue"] < 0.001 else f"p = {row['pvalue']:.2f}".replace(".", ",")
        ax.text(0.03, 0.96, f"β = {row['coef']:.2f}".replace(".", ",").replace("-", "−") + f", {p_text}", transform=ax.transAxes,
                va="top", fontsize=9, color=INK_2)
        ax.set_title(SHORT_TARGETS[target])
        ax.set_xlabel("ln доходов, очищенный от прочих факторов")
        ax.set_ylabel(f"Очищенный показатель, {UNITS[target]}")
    fig.suptitle("Частная регрессия на доходы в панельной модели", x=0.125, ha="left",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.84, wspace=0.28)
    return _save(fig, "partial_regression.png")


def plot_yearly(yearly: pd.DataFrame) -> Path:
    """Коэффициент при ln доходов в регрессиях по отдельным годам."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, target in zip(axes, TARGETS):
        rows = yearly[yearly["target"] == target]
        ax.fill_between(rows["year"], rows["ci_low"], rows["ci_high"], color=BAND, linewidth=0,
                        label="95 %-й интервал")
        ax.plot(rows["year"], rows["coef"], color=SERIES[0], label="Коэффициент")
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_title(SHORT_TARGETS[target])
        ax.set_ylabel(f"{UNITS[target]} на единицу ln доходов")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right")
    fig.suptitle("Коэффициент при доходах в ежегодных регрессиях по регионам", x=0.125, ha="left",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.82, wspace=0.3)
    return _save(fig, "yearly_coefficients.png")


def plot_beta(points: pd.DataFrame, table: pd.DataFrame, first: int, last: int) -> Path:
    """Бета-сходимость: темп роста против исходного уровня."""
    titles = {"income_real": "Реальные доходы", "gdp_real": "Реальный ВРП"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key in zip(axes, titles):
        data = points[points["indicator"] == key]
        row = table[table["indicator"] == key].iloc[0]
        x = data["start"] / 1000
        ax.scatter(x, data["growth_pct"], s=30, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
        grid = np.geomspace(x.min(), x.max(), 100)
        ax.plot(grid, (row["a"] + row["b"] * np.log(grid * 1000)) * 100, color=SERIES[1], linewidth=2)
        ax.set_xscale("log")
        ticks = [t for t in (5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000) if x.min() <= t <= x.max()]
        ax.set_xticks(ticks, labels=[str(t) for t in ticks])
        ax.minorticks_off()
        p_text = "p < 0,001" if row["p"] < 0.001 else f"p = {row['p']:.2f}".replace(".", ",")
        ax.text(1, 1.02, f"b = {row['b']:.4f}, {p_text}, R² = {row['r2']:.2f}".replace(".", ",").replace("-", "−"),
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9, color=INK_2)
        ax.set_title(titles[key])
        ax.set_xlabel(f"Уровень в {first} г., тыс. руб. в ценах 2023 г. (лог. шкала)")
        ax.set_ylabel(f"Среднегодовой рост за {first}–{last} гг., %")
        extreme = data.assign(region=data["region"])
        _label_points(ax, pd.concat([extreme.nlargest(3, "growth_pct"), extreme.nlargest(2, "start")]),
                      "start", "growth_pct", x_scale=1 / 1000)
    fig.suptitle("Бета-сходимость: росли ли бедные регионы быстрее богатых", x=0.125, ha="left",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.85, wspace=0.24)
    return _save(fig, "beta_convergence.png")


def plot_cluster_selection(inertia: pd.DataFrame, scores: pd.DataFrame, chosen: int) -> Path:
    """Выбор числа кластеров: метод локтя и коэффициент силуэта."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, frame, col, title in [(axes[0], inertia, "inertia", "Внутрикластерная сумма квадратов"),
                                  (axes[1], scores, "silhouette", "Средний коэффициент силуэта")]:
        ax.plot(frame["k"], frame[col], color=SERIES[0], marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1)
        picked = frame[frame["k"] == chosen]
        ax.scatter(picked["k"], picked[col], s=90, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Число кластеров k")
        ax.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.24)
    return _save(fig, "cluster_selection.png")


def plot_all(panel, russia, coefs, assignments, cluster_year, fits, extra: dict) -> list[Path]:
    first = int(panel["year"].min())
    return [
        plot_distributions(panel),
        plot_boxplots(panel, cluster_year),
        plot_trends(panel, russia),
        plot_nominal_vs_real(russia),
        plot_convergence(panel),
        plot_beta(extra["beta_points"], extra["beta"], first, cluster_year),
        plot_correlations(panel),
        plot_income_poverty(panel),
        plot_pair_regressions(panel, cluster_year, extra["pairs"], extra["curves"]),
        plot_yearly(extra["yearly"]),
        plot_effects(coefs),
        plot_partial(extra["partials"], coefs),
        plot_diagnostics(fits),
        plot_cluster_selection(extra["inertia"], extra["scores"], assignments["cluster"].nunique()),
        plot_clusters(assignments, cluster_year),
    ]
