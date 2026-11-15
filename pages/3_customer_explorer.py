"""Customer Explorer: one customer's RFM profile, predicted value and purchase history."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.customer_segmentation import SEGMENT_PLAYBOOK
from src.ui import data, theme

def rank_text(share_ahead: float, what: str) -> str:
    """Plain-language rank: share_ahead = fraction of customers who beat this one on the metric."""
    top = max(1, round(share_ahead * 100))
    return f"Top {top}% of customers ({what})" if top <= 50 else f"Bottom {max(1, 100 - top)}% of customers ({what})"



scores, clean, meta = data.load_scores(), data.load_clean(), data.load_json("run_metadata.json")
vm = data.load_json("value_metrics.json")

theme.hero("Customer explorer", "Look up any customer to see their segment, RFM profile, predicted value and order history.", theme.meta_chips(meta))

left, right = st.columns([1, 1.6])
query = left.text_input("Search by CustomerID", placeholder=f"e.g. {int(scores.index[0])}")
ordered = scores.sort_values("monetary", ascending=False).index.tolist()
browse = right.selectbox("...or browse (largest spenders first)", ordered, format_func=lambda i: f"{i}  ·  {scores.at[i, 'segment']}  ·  £{scores.at[i, 'monetary']:,.0f}")
cid = browse
if query.strip():
    if query.strip().isdigit() and int(query) in scores.index:
        cid = int(query)
    else:
        st.warning(f"CustomerID '{query}' was not found. Showing the browse selection instead.")

row = scores.loc[cid]
a, b = st.columns([1.5, 1], gap="large")
with a:
    theme.html(f'<div class="rq-card"><div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px">'
               f'<div><div class="rq-note">Customer</div><div class="rq-big" style="font-size:2rem;font-weight:700">#{cid}</div></div>'
               f'<div>{theme.badge(row["segment"])}</div></div>'
               f'<div style="margin-top:10px">{theme.esc(SEGMENT_PLAYBOOK.get(row["segment"].replace(" II", ""), {}).get("who", ""))}</div>'
               f'<div class="rq-note" style="margin-top:8px">{theme.esc(row["country"])} · customer for {int(row["tenure_days"]):,} days · '
               f'{int(row["active_months"])} active months · {int(row["product_diversity"])} different products</div>'
               f'<div style="margin-top:12px;font-size:.9rem"><b>Suggested action:</b> {theme.esc(SEGMENT_PLAYBOOK.get(row["segment"].replace(" II", ""), {}).get("action", ""))}</div></div>')
with b:
    p = float(row["p_high_value"])
    thr = float(vm["models"][vm["selected_model"]]["validation"]["threshold"])
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number=dict(suffix="%", font=dict(size=34, family="Sora")),
                                   gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 50, 100], ticktext=["0%", "50%", "100%"], tickfont=dict(size=11)), bar=dict(color=theme.INDIGO, thickness=0.28), bgcolor="rgba(127,127,127,.12)", borderwidth=0,
                                              threshold=dict(line=dict(color="#eb6834", width=3), thickness=0.85, value=thr * 100))))
    gauge = theme.style(gauge, 215, legend=False)
    gauge.update_layout(margin=dict(l=44, r=44, t=30, b=0))
    theme.show(gauge)
    st.caption(f"Chance of being a top-20% customer in the next {vm['windows']['horizon_days']} days. Orange line = decision threshold ({thr:.0%}). "
               + ("Predicted **high value**." if p >= thr else "Not predicted high value."))

theme.section("RFM profile", "How this customer compares with all customers.")
theme.kpis([
    {"label": "Recency", "value": f"{int(row['recency_days'])} days", "sub": rank_text((scores["recency_days"] < row["recency_days"]).mean(), "most recent")},
    {"label": "Frequency", "value": f"{int(row['frequency'])} orders", "sub": rank_text((scores["frequency"] > row["frequency"]).mean(), "most frequent")},
    {"label": "Monetary (net)", "value": theme.gbp(row["monetary"], compact=False), "sub": rank_text((scores["monetary"] > row["monetary"]).mean(), "highest spenders")},
    {"label": "Average order", "value": theme.gbp(row["avg_order_value"], compact=False), "sub": f"{row['avg_basket_size']:.0f} products per order"},
])

tx = clean[clean["CustomerID"] == cid]
sales, rets = tx[tx["RecordType"] == "sale"], tx[tx["RecordType"] == "return"]
c1, c2 = st.columns([1.5, 1], gap="large")
with c1:
    theme.section("Spend over time", "Net spend per month (returns subtracted).")
    m = tx.set_index("InvoiceDate")["Revenue"].resample("MS").sum().reset_index()
    fig = go.Figure(go.Bar(x=m["InvoiceDate"], y=m["Revenue"], marker_color=theme.TEAL, hovertemplate="%{x|%b %Y}<br>£%{y:,.0f}<extra></extra>"))
    fig.update_yaxes(tickprefix="£", tickformat="~s")
    theme.show(theme.style(fig, 300, legend=False))
with c2:
    theme.section("Favourite products", "Most units bought.")
    top = sales.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(6).iloc[::-1]
    fig = go.Figure(go.Bar(y=[d.title()[:30] for d in top.index], x=top.values, orientation="h", marker_color=theme.INDIGO, hovertemplate="%{y}<br>%{x:,} units<extra></extra>"))
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=False)
    theme.show(theme.style(fig, 300, legend=False))

theme.section("Purchase history", f"{sales['InvoiceNo'].nunique()} orders and {rets['InvoiceNo'].nunique()} returns. Order values are shown per invoice.")
inv = (tx.groupby(["InvoiceNo", "RecordType"]).agg(Date=("InvoiceDate", "min"), Products=("StockCode", "nunique"), Units=("Quantity", "sum"), Value=("Revenue", "sum"))
       .reset_index().sort_values("Date", ascending=False))
inv["Type"] = inv["RecordType"].map({"sale": "Order", "return": "Return"})
theme.show_df(inv[["InvoiceNo", "Date", "Type", "Products", "Units", "Value"]], height=320, column_config={
    "InvoiceNo": "Invoice", "Date": st.column_config.DatetimeColumn("Date", format="D MMM YYYY, HH:mm"),
    "Value": st.column_config.NumberColumn("Value", format="£%.2f")})
st.caption("CustomerIDs in the source data are anonymised; the dashboard shows no names or addresses.")
