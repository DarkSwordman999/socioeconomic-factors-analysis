"""Полный пересчёт проекта: python -m regions"""

from pathlib import Path

from regions.clusters import cluster_regions
from regions.dataset import LAST_YEAR, PANEL_PATH, build_panel, save_panel
from regions.indicators import TARGETS
from regions.models import effect_of_10pct, fit_spec, run_models, vif_table
from regions.plots import plot_all
from regions.report import build_report

TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"


def main() -> None:
    print("1/5 Сборка панели из data/raw …")
    panel, russia = build_panel()
    save_panel(panel, russia)
    print(f"    {panel['region'].nunique()} регионов, {panel['year'].min()}–{panel['year'].max()}, "
          f"{len(panel)} наблюдений -> {PANEL_PATH.name}")

    print("2/5 Регрессии …")
    coefs, summary = run_models(panel)
    vif = vif_table(panel)
    fits = {target: fit_spec(panel, target, "fe_controls")[0] for target in TARGETS}
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    coefs.round(5).to_csv(TABLES_DIR / "coefficients.csv", index=False)
    summary.round(4).to_csv(TABLES_DIR / "model_summary.csv", index=False)
    effect_of_10pct(coefs).round(4).to_csv(TABLES_DIR / "effects_10pct.csv", index=False)
    vif.round(2).to_csv(TABLES_DIR / "vif.csv", index=False)
    print(summary[["target", "spec", "n", "r2", "r2_within"]].round(3).to_string(index=False))

    print("3/5 Кластеризация …")
    assignments, profiles, scores = cluster_regions(panel, LAST_YEAR)
    assignments.round(3).to_csv(TABLES_DIR / "clusters.csv", index=False)
    profiles.round(2).to_csv(TABLES_DIR / "cluster_profiles.csv", index=False)
    scores.round(4).to_csv(TABLES_DIR / "cluster_silhouette.csv", index=False)
    print(f"    {assignments['cluster'].nunique()} кластера(ов) за {LAST_YEAR} г.")

    print("4/5 Графики …")
    for path in plot_all(panel, russia, coefs, assignments, LAST_YEAR, fits):
        print(f"    {path.name}")

    print("5/5 Отчёт …")
    report = build_report(panel, russia, coefs, summary, vif, fits, assignments, profiles, scores)
    print(f"    {report.relative_to(report.parent.parent)}")


if __name__ == "__main__":
    main()
