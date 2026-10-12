"""Интерактивная панель: streamlit run app.py"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from regions.clusters import cluster_regions
from regions.dataset import LAST_YEAR, PANEL_PATH, RUSSIA_PATH, build_panel, save_panel
from regions.indicators import LABELS, UNITS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SHOWN = ["income_real", "gdp_real", "invest_real", "poverty", "life_exp", "housing",
         "infant_mort", "unemployment", "doctors", "urban", "elderly", "students"]
MONEY = {"income_real", "gdp_real", "invest_real"}

st.set_page_config(page_title="Регионы России", layout="wide")


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PANEL_PATH.exists():
        save_panel(*build_panel())
    return pd.read_csv(PANEL_PATH), pd.read_csv(RUSSIA_PATH)


@st.cache_data
def clusters(panel: pd.DataFrame) -> pd.DataFrame:
    return cluster_regions(panel, LAST_YEAR)[0]


def title(key: str) -> str:
    return f"{LABELS[key]}, {UNITS[key]}"


panel, russia = load()
regions = sorted(panel["region"].unique())

st.title("Социально-экономические показатели регионов России")
st.caption("Данные: Росстат, приложение к сборнику «Регионы России. Социально-экономические показатели», 2025. "
           "Денежные показатели — в ценах 2023 г. с поправкой на уровень цен региона.")

tab_region, tab_compare, tab_rating, tab_types = st.tabs(["Регион", "Связь показателей", "Рейтинг", "Типология"])

with tab_region:
    left, right = st.columns([1, 3])
    chosen = left.multiselect("Регионы", regions, default=["г. Москва", "Республика Тыва"], max_selections=6)
    key = left.selectbox("Показатель", SHOWN, format_func=LABELS.get, key="region_key")
    median = panel.groupby("year")[key].median()

    fig = go.Figure()
    fig.add_scatter(x=median.index, y=median, name="Медианный регион", line=dict(color="#898781", width=2, dash="dot"))
    fig.add_scatter(x=russia["year"], y=russia[key], name="Россия в целом", line=dict(color="#0b0b0b", width=2))
    for region, color in zip(chosen, SERIES):
        data = panel[panel["region"] == region]
        fig.add_scatter(x=data["year"], y=data[key], name=region, line=dict(color=color, width=2))
    fig.update_layout(title=title(key), yaxis_type="log" if key in MONEY else "linear",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.15), height=480)
    right.plotly_chart(fig, width="stretch")

    latest = panel[panel["year"] == LAST_YEAR].set_index("region")
    if chosen:
        table = latest.loc[[r for r in chosen if r in latest.index], SHOWN].T
        table.index = [title(k) for k in table.index]
        st.dataframe(table.round(1), width="stretch")

with tab_compare:
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("По горизонтали", SHOWN, index=0, format_func=LABELS.get)
    y = c2.selectbox("По вертикали", SHOWN, index=3, format_func=LABELS.get)
    year = c3.slider("Год", int(panel["year"].min()), int(panel["year"].max()), LAST_YEAR)
    data = panel[panel["year"] == year].dropna(subset=[x, y])
    fig = px.scatter(data, x=x, y=y, color="district", hover_name="region", log_x=x in MONEY, log_y=y in MONEY,
                     color_discrete_sequence=SERIES, labels={x: title(x), y: title(y), "district": "Федеральный округ"})
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Коэффициент корреляции Спирмена: {data[x].corr(data[y], method='spearman'):.2f} "
               f"({len(data)} регионов)")

with tab_rating:
    c1, c2 = st.columns(2)
    key = c1.selectbox("Показатель", SHOWN, format_func=LABELS.get, key="rating_key")
    year = c2.slider("Год", int(panel["year"].min()), int(panel["year"].max()), LAST_YEAR, key="rating_year")
    data = panel[panel["year"] == year].dropna(subset=[key]).sort_values(key)
    fig = px.bar(data, x=key, y="region", orientation="h", color_discrete_sequence=SERIES,
                 labels={key: title(key), "region": ""})
    fig.update_layout(height=22 * len(data) + 80, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

with tab_types:
    groups = clusters(panel)
    groups["Кластер"] = "Кластер " + groups["cluster"].astype(str)
    fig = px.scatter(groups, x="pc1", y="pc2", color="Кластер", hover_name="region",
                     color_discrete_sequence=SERIES,
                     labels={"pc1": "Первая главная компонента", "pc2": "Вторая главная компонента"})
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(height=520, title=f"Три типа регионов по профилю {LAST_YEAR} г.")
    st.plotly_chart(fig, width="stretch")
    for cluster, group in groups.groupby("cluster"):
        with st.expander(f"Кластер {cluster} — {len(group)} регионов"):
            st.write(", ".join(group["region"]))
