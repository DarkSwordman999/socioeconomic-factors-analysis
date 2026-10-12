"""Полный пересчёт проекта: python -m regions"""

import warnings
from pathlib import Path

from regions import analysis
from regions.clusters import cluster_regions
from regions.dataset import FIRST_YEAR, LAST_YEAR, PANEL_PATH, build_panel, save_panel
from regions.indicators import TARGETS
from regions.models import effect_of_10pct, fit_spec, run_models, vif_table
from regions.plots import plot_all
from regions.report import DESCRIBED, build_report

TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"


def main() -> None:
    warnings.filterwarnings("ignore", message="covariance of constraints")

    print("1/6 Сборка панели из data/raw …")
    panel, russia = build_panel()
    save_panel(panel, russia)
    print(f"    {panel['region'].nunique()} регионов, {panel['year'].min()}–{panel['year'].max()}, "
          f"{len(panel)} наблюдений -> {PANEL_PATH.name}")

    print("2/6 Разведочный анализ, корреляции, парные регрессии …")
    pairs, curves = analysis.pair_models(panel, LAST_YEAR)
    cs_coefs, cs_quality, _ = analysis.cross_section(panel, LAST_YEAR)
    beta, beta_points = analysis.beta_convergence(panel, FIRST_YEAR, LAST_YEAR)
    results = {
        "panel": panel, "russia": russia,
        "normality": analysis.normality(panel, LAST_YEAR, DESCRIBED),
        "extremes": {target: analysis.extremes(panel, LAST_YEAR, target) for target in TARGETS},
        "corr_tests": analysis.corr_tests(panel, LAST_YEAR),
        "pairs": pairs, "cs_coefs": cs_coefs, "cs_quality": cs_quality,
        "yearly": analysis.yearly_coefficients(panel),
        "sigma": analysis.sigma_convergence(panel), "beta": beta,
    }

    print("3/6 Панельные модели …")
    coefs, summary = run_models(panel)
    fits = {target: fit_spec(panel, target, "fe_controls")[0] for target in TARGETS}
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    coefs.round(5).to_csv(TABLES_DIR / "coefficients.csv", index=False)
    summary.round(4).to_csv(TABLES_DIR / "model_summary.csv", index=False)
    effect_of_10pct(coefs).round(4).to_csv(TABLES_DIR / "effects_10pct.csv", index=False)
    vif_table(panel).round(2).to_csv(TABLES_DIR / "vif.csv", index=False)
    print(summary[["target", "spec", "n", "r2", "r2_within"]].round(3).to_string(index=False))

    print("4/6 Кластеризация …")
    assignments, profiles, scores = cluster_regions(panel, LAST_YEAR)
    assignments.round(3).to_csv(TABLES_DIR / "clusters.csv", index=False)
    profiles.round(2).to_csv(TABLES_DIR / "cluster_profiles.csv", index=False)
    scores.round(4).to_csv(TABLES_DIR / "cluster_silhouette.csv", index=False)
    inertia, loadings, explained, kruskal = analysis.cluster_diagnostics(panel, LAST_YEAR, assignments)
    leaders = assignments.loc[assignments["cluster"] == 1, "region"].tolist()
    results |= {
        "coefs": coefs, "summary": summary, "fits": fits,
        "panel_tests": analysis.panel_tests(panel), "elasticity": analysis.poverty_elasticity(panel),
        "robustness": analysis.robustness(panel, leaders),
        "assignments": assignments, "profiles": profiles, "scores": scores,
        "inertia": inertia, "loadings": loadings, "explained": explained, "kruskal": kruskal,
    }
    print(f"    {assignments['cluster'].nunique()} кластера(ов) за {LAST_YEAR} г.")

    print("5/6 Графики …")
    extra = {
        "pairs": pairs, "curves": curves, "yearly": results["yearly"], "beta": beta, "beta_points": beta_points,
        "partials": {target: analysis.partial_regression(panel, target) for target in TARGETS},
        "inertia": inertia, "scores": scores,
    }
    for path in plot_all(panel, russia, coefs, assignments, LAST_YEAR, fits, extra):
        print(f"    {path.name}")

    print("6/6 Отчёт …")
    report = build_report(results)
    print(f"    {report.relative_to(report.parent.parent)}")


if __name__ == "__main__":
    main()
