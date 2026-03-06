import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Data Breach Dashboard", page_icon="🔒", layout="wide",
                   initial_sidebar_state="expanded")

# --- data loading ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "dataset", "raw", "DataBreach_dataset.csv")

# keep consistent colors for impact levels across all tabs
IMPACT_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c", "Critical": "#8e44ad"}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df["Impact Level"] = pd.Categorical(
        df["Impact Level"], categories=["Low", "Medium", "High", "Critical"], ordered=True
    )
    df["Loss Category"] = pd.cut(
        df["Financial Loss"],
        bins=[0, 500_000, 1_000_000, 1_500_000, 2_000_000, float("inf")],
        labels=["<500K", "500K-1M", "1M-1.5M", "1.5M-2M", ">2M"],
    )
    df["Records Category"] = pd.cut(
        df["Records Compromised"],
        bins=[0, 200_000, 400_000, 600_000, 800_000, float("inf")],
        labels=["<200K", "200K-400K", "400K-600K", "600K-800K", ">800K"],
    )
    return df


df = load_data()

# --- sidebar filters ---

st.sidebar.title("Filters")

year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

selected_industries = st.sidebar.multiselect(
    "Industry", sorted(df["Industry"].unique()), default=sorted(df["Industry"].unique())
)
selected_breach_types = st.sidebar.multiselect(
    "Breach Type", sorted(df["Type of Breach"].unique()), default=sorted(df["Type of Breach"].unique())
)
selected_impact = st.sidebar.multiselect(
    "Impact Level", list(IMPACT_COLORS.keys()), default=list(IMPACT_COLORS.keys())
)
selected_errors = st.sidebar.multiselect(
    "Human Error Factor", sorted(df["Human Error Factor"].unique()),
    default=sorted(df["Human Error Factor"].unique())
)
selected_mitigations = st.sidebar.multiselect(
    "Mitigation Measures", sorted(df["Mitigation Measures"].unique()),
    default=sorted(df["Mitigation Measures"].unique())
)

# filter
mask = (
    df["Year"].between(*year_range)
    & df["Industry"].isin(selected_industries)
    & df["Type of Breach"].isin(selected_breach_types)
    & df["Impact Level"].isin(selected_impact)
    & df["Human Error Factor"].isin(selected_errors)
    & df["Mitigation Measures"].isin(selected_mitigations)
)
fdf = df[mask].copy()


def fmt_number(n, prefix=""):
    """Shorten big numbers: 1500000 -> 1.5M, 18000000000 -> 18.0B, etc."""
    abs_n = abs(n)
    sign = "-" if n < 0 else ""
    if abs_n >= 1_000_000_000:
        return f"{sign}{prefix}{abs_n / 1e9:.1f}B"
    if abs_n >= 1_000_000:
        return f"{sign}{prefix}{abs_n / 1e6:.1f}M"
    if abs_n >= 1_000:
        return f"{sign}{prefix}{abs_n / 1e3:.1f}K"
    return f"{sign}{prefix}{abs_n:,.0f}"


# --- header + KPIs ---

st.title("Data Breach Analysis Dashboard")
st.caption(f"Showing **{len(fdf):,}** / **{len(df):,}** records  |  {year_range[0]}\u2013{year_range[1]}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Breaches", f"{len(fdf):,}")
k2.metric("Total Financial Loss", fmt_number(fdf["Financial Loss"].sum(), "$"))
k3.metric("Avg Loss / Breach", fmt_number(fdf["Financial Loss"].mean(), "$") if len(fdf) else "$0")
k4.metric("Records Compromised", fmt_number(fdf["Records Compromised"].sum()))
k5.metric("Companies Affected", fdf["Company Name"].nunique())

st.divider()

# --- tabs ---

tab_overview, tab_breach, tab_industry, tab_trends, tab_error, tab_financial, tab_companies, tab_mitigation, tab_explorer = st.tabs([
    "Overview", "Breach Types", "Industry", "Trends",
    "Human Error", "Financial Impact", "Companies", "Mitigation", "Data Explorer",
])


# ======================= OVERVIEW =======================

with tab_overview:
    st.subheader("Dataset Overview")

    left, right = st.columns(2)

    with left:
        breach_counts = fdf["Type of Breach"].value_counts().reset_index()
        breach_counts.columns = ["Breach Type", "Count"]
        fig = px.pie(breach_counts, names="Breach Type", values="Count",
                     title="Breach Type Distribution", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        ind_counts = fdf["Industry"].value_counts().reset_index()
        ind_counts.columns = ["Industry", "Count"]
        fig = px.pie(ind_counts, names="Industry", values="Count",
                     title="Industry Distribution", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)

    with left2:
        impact_counts = (fdf["Impact Level"].value_counts()
                         .reindex(["Low", "Medium", "High", "Critical"]).reset_index())
        impact_counts.columns = ["Impact Level", "Count"]
        fig = px.bar(impact_counts, x="Impact Level", y="Count", color="Impact Level",
                     title="Impact Level Distribution", color_discrete_map=IMPACT_COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        loss_cat = fdf["Loss Category"].value_counts().sort_index().reset_index()
        loss_cat.columns = ["Loss Category", "Count"]
        fig = px.bar(loss_cat, x="Loss Category", y="Count",
                     title="Financial Loss Buckets", color="Count",
                     color_continuous_scale="OrRd")
        st.plotly_chart(fig, use_container_width=True)


# ======================= BREACH TYPES =======================

with tab_breach:
    st.subheader("Breach Type Analysis")

    left, right = st.columns(2)
    with left:
        freq = fdf["Type of Breach"].value_counts().reset_index()
        freq.columns = ["Breach Type", "Count"]
        fig = px.bar(freq, x="Breach Type", y="Count", color="Breach Type",
                     title="Incidents by Breach Type",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(fdf, x="Type of Breach", y="Financial Loss", color="Type of Breach",
                     title="Financial Loss by Breach Type",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # summary table
    bt_summary = (fdf.groupby("Type of Breach")
                  .agg(Incidents=("Year", "count"),
                       Avg_Loss=("Financial Loss", "mean"),
                       Total_Loss=("Financial Loss", "sum"),
                       Avg_Records=("Records Compromised", "mean"),
                       Total_Records=("Records Compromised", "sum"))
                  .round(2).sort_values("Avg_Loss", ascending=False))
    bt_summary.columns = ["Incidents", "Avg Loss ($)", "Total Loss ($)", "Avg Records", "Total Records"]
    st.dataframe(bt_summary.style.format({
        "Incidents": "{:,}", "Avg Loss ($)": "${:,.2f}", "Total Loss ($)": "${:,.2f}",
        "Avg Records": "{:,.0f}", "Total Records": "{:,}",
    }), use_container_width=True)

    # violin gives a better view of the full distribution shape than box
    fig = px.violin(fdf, x="Type of Breach", y="Records Compromised", color="Type of Breach",
                    box=True, title="Records Compromised Distribution by Breach Type",
                    color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# ======================= INDUSTRY =======================

with tab_industry:
    st.subheader("Industry Analysis")

    left, right = st.columns(2)
    with left:
        ic = fdf["Industry"].value_counts().reset_index()
        ic.columns = ["Industry", "Count"]
        fig = px.bar(ic, y="Industry", x="Count", orientation="h",
                     title="Breaches per Industry", color="Industry",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        il = (fdf.groupby("Industry")["Financial Loss"].mean()
              .sort_values().reset_index())
        il.columns = ["Industry", "Avg Loss"]
        fig = px.bar(il, y="Industry", x="Avg Loss", orientation="h",
                     title="Average Financial Loss by Industry",
                     color="Avg Loss", color_continuous_scale="YlOrRd")
        st.plotly_chart(fig, use_container_width=True)

    # heatmap - wanted to see if some industries attract certain attack types more
    cross = pd.crosstab(fdf["Industry"], fdf["Type of Breach"])
    fig = px.imshow(cross, text_auto=True, color_continuous_scale="YlOrRd",
                    title="Breach Type x Industry", aspect="auto",
                    labels=dict(x="Type of Breach", y="Industry", color="Count"))
    st.plotly_chart(fig, use_container_width=True)

    impact_ind = pd.crosstab(fdf["Industry"], fdf["Impact Level"]).reset_index()
    melted = impact_ind.melt(id_vars="Industry", var_name="Impact Level", value_name="Count")
    fig = px.bar(melted, x="Industry", y="Count", color="Impact Level", barmode="group",
                 title="Impact Level Across Industries", color_discrete_map=IMPACT_COLORS)
    st.plotly_chart(fig, use_container_width=True)

    # detailed table
    ind_tbl = (fdf.groupby("Industry")
               .agg(Breaches=("Year", "count"), Total_Loss=("Financial Loss", "sum"),
                    Avg_Loss=("Financial Loss", "mean"), Total_Records=("Records Compromised", "sum"),
                    Avg_Records=("Records Compromised", "mean"))
               .round(2).sort_values("Total_Loss", ascending=False))
    ind_tbl.columns = ["Breaches", "Total Loss ($)", "Avg Loss ($)", "Total Records", "Avg Records"]
    st.dataframe(ind_tbl.style.format({
        "Breaches": "{:,}", "Total Loss ($)": "${:,.2f}", "Avg Loss ($)": "${:,.2f}",
        "Total Records": "{:,}", "Avg Records": "{:,.0f}",
    }), use_container_width=True)


# ======================= TRENDS =======================

with tab_trends:
    st.subheader("Temporal Trends")

    yearly = fdf.groupby("Year").size().reset_index(name="Count")
    fig = px.line(yearly, x="Year", y="Count", markers=True,
                  title="Breaches Over Time")
    fig.update_traces(line=dict(width=3))
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

    # stacked view: yearly loss + records side by side
    yr_agg = (fdf.groupby("Year")
              .agg(total_loss=("Financial Loss", "sum"), total_records=("Records Compromised", "sum"))
              .reset_index())
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Total Financial Loss by Year",
                                        "Total Records Compromised by Year"))
    fig.add_trace(go.Bar(x=yr_agg["Year"], y=yr_agg["total_loss"],
                         name="Loss ($)", marker_color="indianred"), row=1, col=1)
    fig.add_trace(go.Bar(x=yr_agg["Year"], y=yr_agg["total_records"],
                         name="Records", marker_color="steelblue"), row=2, col=1)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        by_yr = fdf.groupby(["Year", "Type of Breach"]).size().reset_index(name="Count")
        fig = px.area(by_yr, x="Year", y="Count", color="Type of Breach",
                      title="Breach Type Trends")
        fig.update_layout(xaxis=dict(dtick=2))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        iy = fdf.groupby(["Year", "Industry"]).size().reset_index(name="Count")
        fig = px.line(iy, x="Year", y="Count", color="Industry", markers=True,
                      title="Industry Trends Over Time")
        fig.update_layout(xaxis=dict(dtick=2))
        st.plotly_chart(fig, use_container_width=True)

    # year-over-year change
    if len(yearly) > 1:
        yearly["YoY (%)"] = yearly["Count"].pct_change() * 100
        fig = px.bar(yearly.dropna(), x="Year", y="YoY (%)",
                     title="Year-over-Year Change (%)",
                     color="YoY (%)", color_continuous_scale="RdYlGn_r")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


# ======================= HUMAN ERROR =======================

with tab_error:
    st.subheader("Human Error Factor Analysis")

    left, right = st.columns(2)
    with left:
        ec = fdf["Human Error Factor"].value_counts().reset_index()
        ec.columns = ["Factor", "Count"]
        fig = px.bar(ec, x="Factor", y="Count", color="Factor",
                     title="Human Error Frequency",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        el = (fdf.groupby("Human Error Factor")["Financial Loss"]
              .mean().sort_values().reset_index())
        el.columns = ["Factor", "Avg Loss"]
        fig = px.bar(el, y="Factor", x="Avg Loss", orientation="h",
                     title="Avg Loss by Error Factor",
                     color="Avg Loss", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

    # stacked: error factor vs impact
    ei = pd.crosstab(fdf["Human Error Factor"], fdf["Impact Level"]).reset_index()
    ei_melt = ei.melt(id_vars="Human Error Factor", var_name="Impact Level", value_name="Count")
    fig = px.bar(ei_melt, x="Human Error Factor", y="Count", color="Impact Level",
                 barmode="stack", title="Human Error vs Impact Level",
                 color_discrete_map=IMPACT_COLORS)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # heatmap to check if error types cluster in certain industries
    err_ind = pd.crosstab(fdf["Industry"], fdf["Human Error Factor"])
    fig = px.imshow(err_ind, text_auto=True, color_continuous_scale="Blues",
                    title="Error Factors Across Industries", aspect="auto",
                    labels=dict(x="Human Error Factor", y="Industry", color="Count"))
    st.plotly_chart(fig, use_container_width=True)


# ======================= FINANCIAL IMPACT =======================

with tab_financial:
    st.subheader("Financial Impact & Records")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(fdf, x="Financial Loss", nbins=50,
                           title="Financial Loss Distribution",
                           color_discrete_sequence=["coral"])
        fig.add_vline(x=fdf["Financial Loss"].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: ${fdf['Financial Loss'].mean():,.0f}")
        fig.add_vline(x=fdf["Financial Loss"].median(), line_dash="dash", line_color="blue",
                      annotation_text=f"Median: ${fdf['Financial Loss'].median():,.0f}")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(fdf, x="Records Compromised", nbins=50,
                           title="Records Compromised Distribution",
                           color_discrete_sequence=["steelblue"])
        fig.add_vline(x=fdf["Records Compromised"].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {fdf['Records Compromised'].mean():,.0f}")
        fig.add_vline(x=fdf["Records Compromised"].median(), line_dash="dash", line_color="blue",
                      annotation_text=f"Median: {fdf['Records Compromised'].median():,.0f}")
        st.plotly_chart(fig, use_container_width=True)

    # scatter to see if theres any correlation between records and loss
    fig = px.scatter(fdf, x="Records Compromised", y="Financial Loss", color="Industry",
                     size="Records Compromised",
                     hover_data=["Company Name", "Type of Breach", "Year"],
                     title="Records vs Financial Loss", opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        fig = px.violin(fdf, x="Impact Level", y="Financial Loss", color="Impact Level",
                        box=True, points="outliers", title="Loss by Impact Level",
                        category_orders={"Impact Level": list(IMPACT_COLORS.keys())},
                        color_discrete_map=IMPACT_COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        corr = fdf.select_dtypes(include=[np.number]).corr().round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Correlation Matrix", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)


# ======================= COMPANIES =======================

with tab_companies:
    st.subheader("Company-Level Analysis")
    top_n = st.slider("Number of companies to show", 5, 30, 15)

    left, right = st.columns(2)
    with left:
        tc = fdf["Company Name"].value_counts().head(top_n).reset_index()
        tc.columns = ["Company", "Breaches"]
        fig = px.bar(tc, x="Company", y="Breaches",
                     title=f"Top {top_n} Most Breached Companies",
                     color="Breaches", color_continuous_scale="Viridis")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        cl = (fdf.groupby("Company Name")
              .agg(total_loss=("Financial Loss", "sum"), avg_loss=("Financial Loss", "mean"),
                   total_records=("Records Compromised", "sum"), breaches=("Year", "count"))
              .sort_values("total_loss", ascending=False).head(top_n).reset_index())
        fig = px.bar(cl, x="Company Name", y="total_loss", color="breaches",
                     color_continuous_scale="Reds",
                     hover_data=["avg_loss", "total_records", "breaches"],
                     title=f"Top {top_n} by Total Loss",
                     labels={"total_loss": "Total Loss ($)", "breaches": "Breaches"})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # what kinds of attacks hit the top 10 most?
    top10 = fdf["Company Name"].value_counts().head(10).index
    top10_df = fdf[fdf["Company Name"].isin(top10)]
    ct = pd.crosstab(top10_df["Company Name"], top10_df["Type of Breach"]).reset_index()
    ct_melt = ct.melt(id_vars="Company Name", var_name="Breach Type", value_name="Count")
    fig = px.bar(ct_melt, x="Company Name", y="Count", color="Breach Type",
                 title="Attack Profile of Top 10 Companies", barmode="stack",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # full table (scrollable)
    comp_tbl = (fdf.groupby("Company Name")
                .agg(Breaches=("Year", "count"), Total_Loss=("Financial Loss", "sum"),
                     Avg_Loss=("Financial Loss", "mean"), Total_Records=("Records Compromised", "sum"))
                .round(2).sort_values("Total_Loss", ascending=False))
    comp_tbl.columns = ["Breaches", "Total Loss ($)", "Avg Loss ($)", "Total Records"]
    st.dataframe(comp_tbl.style.format({
        "Breaches": "{:,}", "Total Loss ($)": "${:,.2f}",
        "Avg Loss ($)": "${:,.2f}", "Total Records": "{:,}",
    }), use_container_width=True)


# ======================= MITIGATION =======================

with tab_mitigation:
    st.subheader("Mitigation Measures")

    left, right = st.columns(2)
    with left:
        mc = fdf["Mitigation Measures"].value_counts().reset_index()
        mc.columns = ["Measure", "Count"]
        fig = px.bar(mc, x="Measure", y="Count", color="Measure",
                     title="How Often Each Measure Appears",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        ml = (fdf.groupby("Mitigation Measures")["Financial Loss"]
              .mean().sort_values().reset_index())
        ml.columns = ["Measure", "Avg Loss"]
        fig = px.bar(ml, y="Measure", x="Avg Loss", orientation="h",
                     title="Avg Loss by Mitigation Measure",
                     color="Avg Loss", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

    mb = pd.crosstab(fdf["Mitigation Measures"], fdf["Type of Breach"])
    fig = px.imshow(mb, text_auto=True, color_continuous_scale="Greens",
                    title="Mitigation Measures vs Breach Types", aspect="auto",
                    labels=dict(x="Type of Breach", y="Mitigation Measure", color="Count"))
    st.plotly_chart(fig, use_container_width=True)

    # sunburst: hierarchical view of mitigation -> impact
    mi = (fdf.groupby(["Mitigation Measures", "Impact Level"])
          .agg(avg_loss=("Financial Loss", "mean"), count=("Year", "count"))
          .round(2).reset_index())
    fig = px.sunburst(mi, path=["Mitigation Measures", "Impact Level"], values="count",
                      color="avg_loss", color_continuous_scale="RdYlGn_r",
                      title="Mitigation → Impact Breakdown (color = avg loss)",
                      labels={"avg_loss": "Avg Loss ($)"})
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # what proportion of each impact level does each measure have?
    mi_pct = pd.crosstab(fdf["Mitigation Measures"], fdf["Impact Level"],
                         normalize="index").round(4) * 100
    mi_pct = mi_pct.reset_index()
    mi_melt = mi_pct.melt(id_vars="Mitigation Measures", var_name="Impact Level",
                           value_name="Pct")
    fig = px.bar(mi_melt, x="Mitigation Measures", y="Pct", color="Impact Level",
                 title="Impact Level Proportions by Measure (%)", barmode="stack",
                 color_discrete_map=IMPACT_COLORS)
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)


# ======================= DATA EXPLORER =======================

with tab_explorer:
    st.subheader("Data Explorer")
    st.write("Browse the filtered dataset. Use the search box to look up specific companies.")

    search = st.text_input("Search company name")
    show_df = fdf.copy()
    if search:
        show_df = show_df[show_df["Company Name"].str.contains(search, case=False, na=False)]

    st.dataframe(
        show_df.drop(columns=["Loss Category", "Records Category"], errors="ignore"),
        use_container_width=True, height=500,
    )

    st.download_button("Download filtered data (CSV)",
                       show_df.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_breach_data.csv", mime="text/csv")

    st.subheader("Summary Statistics")
    st.dataframe(fdf[["Financial Loss", "Records Compromised", "Year"]].describe().round(2),
                 use_container_width=True)


# --- footer ---
st.divider()
st.caption("Data Breach Dashboard  |  12,378 records (2000-2024)  |  Streamlit + Plotly")
