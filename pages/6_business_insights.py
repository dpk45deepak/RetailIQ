"""Business Insights: revenue trends, retention opportunities and segment-level recommendations."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import eda
from src.analytics.business_insights import retention_opportunities
from src.models.customer_segmentation import SEGMENT_PLAYBOOK
from src.ui import data, theme

scores, clean, meta = data.load_scores(), data.load_clean(), data.load_json("run_metadata.json")
profile, insights, affinity = data.load_profile(), data.load_json("business_insights.json"), data.load_affinity()

theme.hero("Business insights", "Findings computed from your data and models, each with a concrete next step.", theme.meta_chips(meta))

theme.section("Key findings", "Generated from the current data. Nothing here is hard-coded.")
theme.html('<div class="rq-grid">' + "".join(theme.insight_card(i["kind"], i["title"], i["finding"], i["action"]) for i in insights) + "</div>")

theme.section("Revenue trend and seasonality")
m = eda.monthly_revenue(clean)
m["year"], m["moy"] = m["month"].dt.year, m["month"].dt.month
fig = go.Figure()
palette = [theme.BLUE, "#1baf7a", "#eda100"]
for i, (yr, g) in enumerate(m.groupby("year")):
    fig.add_scatter(x=[pd.Timestamp(2000, mm, 1).strftime("%b") for mm in g["moy"]], y=g["net_revenue"], mode="lines+markers", name=str(yr),
                    line=dict(width=2.5, color=palette[i % 3]), marker=dict(size=7, line=dict(color="rgba(255,255,255,.9)", width=1.5)),
                    hovertemplate=f"{yr} " + "%{x}<br>£%{y:,.0f}<extra></extra>")
fig.update_yaxes(tickprefix="£", tickformat="~s")
theme.show(theme.style(fig, 340))
st.caption("Years overlaid so seasonal peaks line up. The first and last years may be partial.")

theme.section("Revenue contribution by segment", "Who generates the money, and how recently they were active.")
names = theme.ordered_segments(profile.index)
a, b = st.columns([1.2, 1], gap="large")
with a:
    fig = go.Figure(go.Bar(x=names, y=[profile.loc[n, "revenue"] for n in names], marker_color=[theme.seg_color(n) for n in names],
                           text=[f"{profile.loc[n, 'revenue_share']:.0%}" for n in names], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>"))
    fig.update_yaxes(tickprefix="£", tickformat="~s", showgrid=True)
    fig.update_xaxes(tickfont=dict(size=11))
    theme.show(theme.style(fig, 330, legend=False))
with b:
    p = scores.assign(bucket=pd.cut(scores["recency_days"], [-1, 30, 90, 180, 365, 10_000], labels=["0-30 days", "31-90", "91-180", "181-365", "365+"]))
    g = p.groupby("bucket", observed=False)["monetary"].sum().clip(lower=0)
    fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.values, marker_color=theme.INDIGO, hovertemplate="%{x} since last order<br>£%{y:,.0f} historical spend<extra></extra>"))
    fig.update_yaxes(tickprefix="£", tickformat="~s", showgrid=True)
    theme.show(theme.style(fig, 330, legend=False, title="Historical spend by time since last order"))

theme.section("Customer retention opportunities", "Largest at-risk and hibernating customers by historical spend: the first win-back list.")
ro = retention_opportunities(scores, n=200)
f1, f2 = st.columns([1.3, 1])
seg_filter = f1.multiselect("Segments", ["At-Risk Customers", "Hibernating Customers"], default=["At-Risk Customers"])
min_spend = f2.slider("Minimum historical spend (£)", 0, int(max(ro["monetary"].max(), 1)), 0, step=250)
ro = ro[ro["segment"].isin(seg_filter) & (ro["monetary"] >= min_spend)]
theme.kpis([{"label": "Customers on the list", "value": f"{len(ro):,}"}, {"label": "Historical spend at stake", "value": theme.gbp(ro["monetary"].sum())},
            {"label": "Median days silent", "value": f"{ro['recency_days'].median():.0f}" if len(ro) else "-"}])
view = ro.reset_index().rename(columns={"CustomerID": "Customer"})
theme.show_df(view[["Customer", "segment", "recency_days", "frequency", "monetary", "p_high_value", "country"]], height=330, column_config={
    "segment": "Segment", "recency_days": st.column_config.NumberColumn("Days since order", format="%d"), "frequency": st.column_config.NumberColumn("Orders", format="%d"),
    "monetary": st.column_config.NumberColumn("Historical spend", format="£%.0f"), "p_high_value": st.column_config.ProgressColumn("Chance high-value next quarter", format="percent", min_value=0, max_value=1),
    "country": "Country"})
st.download_button("Download win-back list (CSV)", view.to_csv(index=False).encode(), "winback_list.csv", "text/csv", type="primary")

theme.section("Segment-level recommendations")
cols = st.columns(2, gap="large")
for i, name in enumerate(names):
    play = SEGMENT_PLAYBOOK.get(name.replace(" II", ""), {"who": "", "action": ""})
    r = profile.loc[name]
    with cols[i % 2]:
        theme.html(f'<div class="rq-card" style="margin-bottom:14px">{theme.badge(name)}<div style="margin:10px 0 4px;font-size:.9rem">{theme.esc(play["who"])} '
                   f'<span class="rq-note">{int(r.customers):,} customers · {r.revenue_share:.0%} of revenue</span></div><div style="font-size:.9rem"><b>Do this:</b> {theme.esc(play["action"])}</div></div>')

theme.section("Product affinity", "Pairs bought together more often than chance. Lift above 1 means a real association.")
aff = affinity.head(12).copy()
aff["Pair"] = aff["description_a"].str.title().str[:28] + "  +  " + aff["description_b"].str.title().str[:28]
fig = go.Figure(go.Bar(y=aff["Pair"].iloc[::-1], x=aff["lift"].iloc[::-1], orientation="h", marker_color=theme.TEAL, customdata=aff[["orders_together", "confidence_a_to_b"]].iloc[::-1],
                       hovertemplate="%{y}<br>lift %{x:.1f}x · %{customdata[0]} orders together<br>%{customdata[1]:.0%} of orders with the first include the second<extra></extra>"))
fig.update_xaxes(title="Lift (x more likely than chance)", showgrid=True)
fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
theme.show(theme.style(fig, 420, legend=False))
