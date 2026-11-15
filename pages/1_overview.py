"""Overview: headline KPIs, revenue trend, segment mix, geography, products and buying rhythm."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import eda
from src.ui import data, theme
from src.utils.config import SEGMENT_ORDER

clean, scores, meta = data.load_clean(), data.load_scores(), data.load_json("run_metadata.json")
dmin, dmax = clean["InvoiceDate"].min().date(), clean["InvoiceDate"].max().date()

theme.hero("Customer intelligence overview",
           f"Revenue, customers and products from {dmin:%b %Y} to {dmax:%b %Y}. Revenue is net of returns.", theme.meta_chips(meta))

picked = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, help="Filters the KPIs and charts on this page.")
start, end = (picked if isinstance(picked, (tuple, list)) and len(picked) == 2 else (dmin, dmax))
c = clean[(clean["InvoiceDate"] >= pd.Timestamp(start)) & (clean["InvoiceDate"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
if c.empty:
    st.info("No transactions in this date range. Pick a wider range.")
    st.stop()

k = eda.kpis(c)
theme.kpis([
    {"label": "Total customers", "value": f"{k['customers']:,}", "sub": "with at least one order"},
    {"label": "Net revenue", "value": theme.gbp(k["net_revenue"]), "sub": f"{theme.gbp(k['returns_value'])} returned ({k['return_rate']:.1%})"},
    {"label": "Average order value", "value": theme.gbp(k["avg_order_value"], compact=False), "sub": f"{k['transactions']:,} orders"},
    {"label": "Total products", "value": f"{k['products']:,}", "sub": f"sold in {k['countries']} countries"},
])

left, right = st.columns([2, 1], gap="large")
with left:
    theme.section("Monthly net revenue", "The last month may be partial.")
    m = eda.monthly_revenue(c)
    fig = go.Figure(go.Scatter(x=m["month"], y=m["net_revenue"], mode="lines", line=dict(color=theme.BLUE, width=2.5, shape="spline", smoothing=0.5),
                               fill="tozeroy", fillcolor="rgba(37,99,235,.12)", name="Net revenue", hovertemplate="%{x|%b %Y}<br>£%{y:,.0f}<extra></extra>"))
    peak = m.loc[m["net_revenue"].idxmax()]
    fig.add_annotation(x=peak["month"], y=peak["net_revenue"], text=f"Peak {peak['month']:%b %Y}: {theme.gbp(peak['net_revenue'])}", showarrow=True,
                       arrowhead=0, ax=-70, ay=-26, font=dict(size=12), arrowcolor="rgba(120,120,120,.6)")
    fig.update_yaxes(tickprefix="£", tickformat="~s")
    theme.show(theme.style(fig, 340, legend=False))
with right:
    theme.section("Customer segments", "Share of customers.")
    seg_counts = scores["segment"].value_counts()
    order = theme.ordered_segments(seg_counts.index)
    pie = go.Figure(go.Pie(labels=order, values=[seg_counts[s] for s in order], hole=0.66, sort=False, direction="clockwise",
                           marker=dict(colors=[theme.seg_color(s) for s in order], line=dict(color="rgba(0,0,0,0)", width=2)),
                           textinfo="percent", textfont=dict(size=12, color="#fff"), hovertemplate="%{label}<br>%{value:,} customers (%{percent})<extra></extra>"))
    pie.add_annotation(text=f"<b>{len(scores):,}</b><br>customers", showarrow=False, font=dict(size=13))
    pie = theme.style(pie, 340)
    pie.update_layout(legend=dict(orientation="h", y=-0.04, x=0.5, xanchor="center", yanchor="top", font=dict(size=11)), margin=dict(l=0, r=0, t=6, b=44))
    theme.show(pie)

a, b = st.columns(2, gap="large")
with a:
    theme.section("Revenue by country", "Top 8 markets by net revenue.")
    cr = eda.country_revenue(c).head(8).iloc[::-1]
    fig = go.Figure(go.Bar(y=cr["Country"], x=cr["net_revenue"], orientation="h", marker_color=theme.TEAL, text=[f"{s:.1%}" for s in cr["share"]],
                           textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>£%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(tickprefix="£", tickformat="~s", showgrid=True)
    fig.update_yaxes(showgrid=False)
    theme.show(theme.style(fig, 330, legend=False))
with b:
    theme.section("Top 10 products", "By net revenue.")
    tp = eda.top_products(c, 10).iloc[::-1]
    fig = go.Figure(go.Bar(y=[d.title()[:32] for d in tp["Description"]], x=tp["revenue"], orientation="h", marker_color=theme.INDIGO,
                           customdata=tp[["StockCode", "units"]], hovertemplate="%{y}<br>£%{x:,.0f} · %{customdata[1]:,} units<br>Code %{customdata[0]}<extra></extra>"))
    fig.update_xaxes(tickprefix="£", tickformat="~s", showgrid=True)
    fig.update_yaxes(showgrid=False)
    theme.show(theme.style(fig, 330, legend=False))

theme.section("When customers buy", "Orders by weekday and hour of day: use it to time campaigns and staffing.")
inv = eda.weekday_hour_pattern(c)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
grid = inv.pivot_table(index="weekday", columns="hour", values="InvoiceNo", aggfunc="count").reindex(days).fillna(0)
hm = go.Figure(go.Heatmap(z=grid.values, x=[f"{h}:00" for h in grid.columns], y=[d[:3] for d in grid.index], colorscale=[[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]],
                          hovertemplate="%{y} %{x}<br>%{z:,.0f} orders<extra></extra>", xgap=3, ygap=3, colorbar=dict(title="Orders", thickness=10, len=0.9)))
hm.update_yaxes(autorange="reversed", showgrid=False)
theme.show(theme.style(hm, 300, legend=False))
