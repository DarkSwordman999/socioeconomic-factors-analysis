# Каталог результатов

Всё в каталоге `results` создаётся командой `python -m regions`. Здесь описано, что находится в каждом файле и какая функция его формирует.

## Отчёт

[`results/REPORT.md`](../results/REPORT.md) — текст исследования. Собирается функцией `regions.report.build_report`.

## Графики: `results/figures`

| Файл | Что показано | Функция в `regions/plots.py` | Раздел отчёта |
|---|---|---|---|
| `nominal_vs_real.png` | доходы России в текущих и реальных ценах, 2000 = 100 | `plot_nominal_vs_real` | 2.2 |
| `distributions.png` | гистограммы шести основных показателей | `plot_distributions` | 3.1 |
| `trends.png` | динамика социальных показателей: медианный регион, разброс 80 % регионов, Россия в целом | `plot_trends` | 3.3 |
| `boxplots.png` | диаграммы размаха социальных показателей по федеральным округам | `plot_boxplots` | 3.4 |
| `convergence.png` | коэффициент вариации доходов и ВРП между регионами по годам | `plot_convergence` | 4.1 |
| `beta_convergence.png` | темп роста за период против исходного уровня, линия регрессии | `plot_beta` | 4.2 |
| `income_poverty.png` | реальные доходы и бедность в 2005 и 2023 годах | `plot_income_poverty` | 5.2 |
| `correlations.png` | матрица корреляций двенадцати показателей (внутри года) | `plot_correlations` | 5.3 |
| `pair_regressions.png` | поля корреляции 3×3 с линейной и лучшей формой парной регрессии | `plot_pair_regressions` | 6.2 |
| `yearly_coefficients.png` | коэффициент при доходах в ежегодных регрессиях с доверительным интервалом | `plot_yearly` | 7.4 |
| `effects.png` | эффект роста фактора на 10 %: наивная модель против панельной | `plot_effects` | 8.5 |
| `partial_regression.png` | частные графики регрессии на доходы в панельной модели | `plot_partial` | 8.5 |
| `diagnostics.png` | факт и прогноз, остатки против прогноза, Q-Q графики остатков | `plot_diagnostics` | 8.8 |
| `cluster_selection.png` | метод локтя и коэффициент силуэта | `plot_cluster_selection` | 9.2 |
| `clusters.png` | регионы на плоскости двух главных компонент, цвет — кластер | `plot_clusters` | 9.4 |

## Таблицы: `results/tables`

Все файлы — CSV в кодировке UTF-8.

### Разведочный анализ

| Файл | Строка — это | Столбцы | Функция |
|---|---|---|---|
| `descriptive_stats.csv` | показатель | `n`, `mean`, `median`, `std`, `min`, `max`, `cv_pct`, `skew`, `kurtosis` | `report.describe` |
| `normality.csv` | показатель | статистика Шапиро — Уилка `w`, `p` и асимметрия — в уровнях и в логарифмах (`*_log`) | `analysis.normality` |

### Сходимость

| Файл | Строка — это | Столбцы | Функция |
|---|---|---|---|
| `sigma_convergence.csv` | год | коэффициент вариации (%) по каждому показателю | `analysis.sigma_convergence` |
| `beta_convergence.csv` | показатель | `a`, `b`, `se`, `p`, `r2`, скорость `speed_pct`, `half_life` | `analysis.beta_convergence` |

### Корреляции

| Файл | Строка — это | Столбцы | Функция |
|---|---|---|---|
| `correlation_tests.csv` | пара «показатель — фактор», срез последнего года | `pearson`, `t`, `p`, `spearman`, `p_spearman` | `analysis.corr_tests` |
| `correlations.csv` | матрица 12×12 | корреляции отклонений от среднего за год | `report.within_year_corr` |

### Регрессии на срезе

| Файл | Строка — это | Столбцы | Функция |
|---|---|---|---|
| `pair_models.csv` | пара × форма связи (45 строк) | параметры `a`, `b`, `c`; `r2`, `r2_adj`, `f`, `p`, `mape`, `elasticity`; `best` — лучшая форма для пары | `analysis.pair_models` |
| `cross_section_coefficients.csv` | показатель × фактор | `coef`, `se`, `t`, `p`, `p_hc3`, `beta_std`, `elasticity`, `vif` | `analysis.cross_section` |
| `cross_section_quality.csv` | показатель | `r2`, `r2_adj`, `f`, `f_p`, `se_reg`, p-значения тестов `bp_p`, `jb_p`, `reset_p`, `dw`, `mape` | `analysis.cross_section` |
| `yearly_coefficients.csv` | год × показатель | коэффициент при ln доходов, границы интервала, `r2` | `analysis.yearly_coefficients` |

### Панельные модели

| Файл | Строка — это | Столбцы | Функция |
|---|---|---|---|
| `coefficients.csv` | показатель × модель × переменная | `coef`, `se`, `pvalue`, `ci_low`, `ci_high` | `models.run_models` |
| `model_summary.csv` | показатель × модель | `n`, `regions`, `r2`, `r2_within` | `models.run_models` |
| `effects_10pct.csv` | показатель × модель × фактор | эффект роста фактора на 10 % и его интервал | `models.effect_of_10pct` |
| `vif.csv` | фактор | коэффициент вздутия дисперсии по всей панели | `models.vif_table` |
| `panel_tests.csv` | показатель | F-тест эффектов `f_fe`, степени свободы, `bp_p`, автокорреляция остатков `rho` | `analysis.panel_tests` |
| `poverty_elasticity.csv` | период | эластичность бедности по доходам, ошибка, интервал | `analysis.poverty_elasticity` |
| `robustness.csv` | вариант × показатель | коэффициент при ln доходов, ошибка, `p`, число наблюдений и регионов | `analysis.robustness` |
| `residual_tests.csv` | показатель | стандартное отклонение, асимметрия и эксцесс остатков, тест Харке — Бера | `report.residual_tests` |

### Типология

| Файл | Строка — это | Столбцы | Функция |
|---|---|---|---|
| `clusters.csv` | регион | округ, номер кластера, девять показателей, координаты `pc1`, `pc2` | `clusters.cluster_regions` |
| `cluster_profiles.csv` | кластер | число регионов и средние значения показателей | `clusters.cluster_regions` |
| `cluster_silhouette.csv` | число кластеров k | средний коэффициент силуэта | `clusters.cluster_regions` |
| `cluster_inertia.csv` | число кластеров k | внутрикластерная сумма квадратов | `analysis.cluster_diagnostics` |
| `pca_loadings.csv` | показатель | нагрузки на три главные компоненты | `analysis.cluster_diagnostics` |
| `cluster_kruskal.csv` | показатель | статистика `h` и `p` критерия Краскела — Уоллиса | `analysis.cluster_diagnostics` |

## Обозначения в таблицах

| Имя | Значение |
|---|---|
| `target` | зависимая переменная: `poverty`, `life_exp`, `housing` |
| `spec` | спецификация панельной модели: `naive`, `real`, `fe`, `fe_controls` |
| `term` | переменная модели: `ln_gdp`, `ln_invest`, `ln_income`, контрольные, `year` |
| `form` | форма парной регрессии: `linear`, `log`, `power`, `exp`, `quadratic` |
| `r2_within` | внутригрупповой R² — доля вариации, объяснённая сверх фиксированных эффектов |
| `p_hc3` | p-значение с ошибками, устойчивыми к гетероскедастичности |
