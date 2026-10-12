"""Полный пересчёт проекта: python -m regions"""

from pathlib import Path

from regions.clusters import cluster_regions
from regions.dataset import LAST_YEAR, PANEL_PATH, build_panel, save_panel
from regions.models import effect_of_10pct, run_models, vif_table
from regions.plots import plot_all

TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"


def main() -> None:
    print("1/4 Сборка панели из data/raw …")
    panel, russia = build_panel()
    save_panel(panel, russia)
    print(f"    {panel['region'].nunique()} регионов, {panel['year'].min()}–{panel['year'].max()}, "
          f"{len(panel)} наблюдений -> {PANEL_PATH.name}")

    print("2/4 Регрессии …")
    coefs, summary = run_models(panel)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    coefs.round(5).to_csv(TABLES_DIR / "coefficients.csv", index=False)
    summary.round(4).to_csv(TABLES_DIR / "model_summary.csv", index=False)
    effect_of_10pct(coefs).round(4).to_csv(TABLES_DIR / "effects_10pct.csv", index=False)
    vif_table(panel).round(2).to_csv(TABLES_DIR / "vif.csv", index=False)
    print(summary[["target", "spec", "n", "r2", "r2_within"]].round(3).to_string(index=False))

    print("3/4 Кластеризация …")
    assignments, profiles, scores = cluster_regions(panel, LAST_YEAR)
    assignments.round(3).to_csv(TABLES_DIR / "clusters.csv", index=False)
    profiles.round(2).to_csv(TABLES_DIR / "cluster_profiles.csv", index=False)
    scores.round(4).to_csv(TABLES_DIR / "cluster_silhouette.csv", index=False)
    print(f"    {assignments['cluster'].nunique()} кластера(ов) за {LAST_YEAR} г.")

    print("4/4 Графики …")
    for path in plot_all(panel, russia, coefs, assignments, LAST_YEAR):
        print(f"    {path.name}")


if __name__ == "__main__":
    main()
