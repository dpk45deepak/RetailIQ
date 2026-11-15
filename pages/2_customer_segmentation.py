"""Customer Segmentation: where each K-Means cluster sits, how big it is and what it is worth."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.customer_segmentation import SEGMENT_PLAYBOOK, segment_profile
from src.ui import data, theme

scores, meta, bundle = data.load_scores(), data.load_json("run_metadata.json"), data.load_seg_bundle()

theme.hero("Customer segmentation",
           "K-Means groups customers by how recently, how often and how much they buy. Filter below to focus on a slice of the base.",
           theme.meta_chips(meta) + [(f"Silhouette {bundle['silhouette']:.2f}", False)])

with st.container():
    f1, f2, f3, f4 = st.columns([2.2, 1.2, 1.2, 1.2])
    segs = theme.ordered_segments(scores["segment"].unique())
    chosen = f1.multiselect("Segments", segs, default=segs)
    countries = ["All countries"] + sorted(scores["country"].unique())
    country = f2.selectbox("Country", countries)
    r_lo, r_hi = f3.slider("Days since last order", 0, int(scores["recency_days"].max()), (0, int(scores["recency_days"].max())))
    m_cap = int(scores["monetary"].quantile(0.99))
    m_lo, m_hi = f4.slider("Net spend (£)", 0, m_cap, (0, m_cap), step=max(m_cap // 100, 1), help="Upper end is capped at the 99th percentile.")

mask = scores["segment"].isin(chosen) & scores["recency_days"].between(r_lo, r_hi)
mask &= (scores["monetary"] >= m_lo) | (m_lo <= 0)   # customers whose returns exceed purchases have net spend < 0
mask &= (scores["monetary"] <= m_hi) | (m_hi >= m_cap)
if country != "All countries":
    mask &= scores["country"] == country
view = scores[mask]
if view.empty:
    st.info("No customers match these filters. Widen the ranges or pick more segments.")
    st.stop()

total_rev = scores["monetary"].clip(lower=0).sum()
theme.kpis([
    {"label": "Customers in view", "value": f"{len(view):,}", "sub": f"{len(view) / len(scores):.0%} of all customers"},
    {"label": "Share of revenue", "value": f"{view['monetary'].clip(lower=0).sum() / total_rev:.0%}", "sub": theme.gbp(view["monetary"].sum())},
    {"label": "Median days since order", "value": f"{view['recency_days'].median():.0f}"},
    {"label": "Median orders", "value": f"{view['frequency'].median():.0f}", "sub": f"median spend {theme.gbp(view['monetary'].median(), compact=False)}"},
])

theme.section("Segment map (PCA)", "Each dot is a customer, flattened from 3 behavioural dimensions to 2. Larger outlined dots are segment centres.")
var = bundle["pca"].explained_variance_ratio_
fig = go.Figure()
order = theme.ordered_segments(view["segment"].unique())
for s in order:
    d = view[view["segment"] == s]
    fig.add_trace(go.Scattergl(x=d["pc1"], y=d["pc2"], mode="markers", name=s, marker=dict(size=6, color=theme.seg_color(s), opacity=0.7),
                               customdata=np.c_[d.index, d["recency_days"], d["frequency"], d["monetary"]],
                               hovertemplate="Customer %{customdata[0]}<br>" + s + "<br>Last order %{customdata[1]:.0f} days ago<br>"
                                             "%{customdata[2]:.0f} orders · £%{customdata[3]:,.0f}<extra></extra>"))
for s in order:
    d = view[view["segment"] == s]
    fig.add_trace(go.Scatter(x=[d["pc1"].mean()], y=[d["pc2"].mean()], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=15, color=theme.seg_color(s), line=dict(color="#fff", width=2.5))))
fig.update_xaxes(title=f"Component 1 ({var[0]:.0%} of variance)", showgrid=True, zeroline=False)
fig.update_yaxes(title=f"Component 2 ({var[1]:.0%} of variance)")
theme.show(theme.style(fig, 470))

prof = segment_profile(scores, scores["segment"])
c1, c2 = st.columns(2, gap="large")
with c1:
    theme.section("Customers vs revenue", "Small groups often carry most of the money.")
    names = theme.ordered_segments(prof.index)
    fig = go.Figure()
    fig.add_bar(name="% of customers", x=names, y=[prof.loc[n, "customer_share"] for n in names], marker_color=theme.BLUE)
    fig.add_bar(name="% of revenue", x=names, y=[prof.loc[n, "revenue_share"] for n in names], marker_color="#1baf7a")
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_traces(texttemplate="%{y:.0%}", textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>")
    fig.update_yaxes(tickformat=".0%", showgrid=True)
    fig.update_xaxes(tickangle=0, tickfont=dict(size=11))
    theme.show(theme.style(fig, 360))
with c2:
    theme.section("Segment characteristics", "Typical (median) customer per segment, compared with the average of all segments.")
    cols = {"recency_median": "Days since order", "frequency_median": "Orders", "monetary_median": "Net spend", "avg_order_value": "Avg order value", "tenure_median": "Tenure (days)"}
    p = prof.loc[names, list(cols)]
    z = (p - p.mean()) / p.std(ddof=0).replace(0, 1)
    text = p.round(0).astype(int).map(lambda v: f"{v:,}").values
    hm = go.Figure(go.Heatmap(z=z.values, x=list(cols.values()), y=names, text=text, texttemplate="%{text}", zmin=-2, zmax=2, xgap=3, ygap=3,
                              colorscale=[[0, "#2a78d6"], [0.5, "#e9e8e4"], [1, "#e34948"]], showscale=False,
                              hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    hm.update_yaxes(autorange="reversed", showgrid=False)
    hm.update_xaxes(side="top", tickfont=dict(size=11))
    theme.show(theme.style(hm, 360, legend=False))
    st.caption("Blue = below the cross-segment average, red = above. Numbers are medians.")

theme.section("Segment playbook", "What each group looks like and what to do about it.")
table = prof.reset_index()
table["Who they are"] = table["segment"].map(lambda s: SEGMENT_PLAYBOOK.get(s.replace(" II", ""), {}).get("who", ""))
table["Recommended action"] = table["segment"].map(lambda s: SEGMENT_PLAYBOOK.get(s.replace(" II", ""), {}).get("action", ""))
table = table.sort_values("revenue", ascending=False)[["segment", "customers", "customer_share", "revenue_share", "recency_median", "frequency_median", "monetary_median", "Who they are", "Recommended action"]]
theme.show_df(table, column_config={
    "segment": "Segment", "customers": st.column_config.NumberColumn("Customers", format="%d"),
    "customer_share": st.column_config.ProgressColumn("Customers %", format="percent", min_value=0, max_value=1),
    "revenue_share": st.column_config.ProgressColumn("Revenue %", format="percent", min_value=0, max_value=1),
    "recency_median": st.column_config.NumberColumn("Median days since order", format="%d"),
    "frequency_median": st.column_config.NumberColumn("Median orders", format="%d"),
    "monetary_median": st.column_config.NumberColumn("Median spend", format="£%d")})

with st.expander("How many segments? Elbow and silhouette"):
    km = pd.DataFrame(bundle["k_metrics"])
    e1, e2 = st.columns(2)
    for col, ycol, title in ((e1, "inertia", "Elbow: within-cluster spread (inertia)"), (e2, "silhouette", "Silhouette score (higher is better)")):
        fig = go.Figure(go.Scatter(x=km["k"], y=km[ycol], mode="lines+markers", line=dict(color=theme.INDIGO, width=2.5), marker=dict(size=9, color=theme.INDIGO)))
        fig.add_vline(x=bundle["k"], line_dash="dot", line_color="rgba(120,120,120,.7)", annotation_text=f"chosen k={bundle['k']}", annotation_position="top right")
        fig.update_xaxes(title="Number of clusters (k)", dtick=1)
        with col:
            theme.show(theme.style(fig, 280, legend=False, title=title))
    st.caption("k is the best silhouette between 3 and 6 clusters; near-ties prefer the larger k for a more actionable split. Six is a ceiling, not a target.")
