"""Product Recommendations: pick a customer, see ranked products, scores and the reason for each."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.recommendations.hybrid import STRATEGIES, WEIGHTS
from src.ui import data, theme

scores, meta = data.load_scores(), data.load_json("run_metadata.json")
reco = data.load_recommender()
ev = data.load_json("recommender_eval.json")

theme.hero("Product recommendations",
           "A hybrid of 'customers like you', 'products like yours' and best-sellers. Products a customer already bought are never shown.",
           theme.meta_chips(meta) + [(f"Precision@5 {ev['models']['hybrid']['precision@5']:.1%}", False)])

c1, c2, c3 = st.columns([2.2, 1.2, 1])
ordered = scores.sort_values("monetary", ascending=False).index.tolist()
cid = c1.selectbox("Customer", ordered, format_func=lambda i: f"{i}  ·  {scores.at[i, 'segment']}  ·  £{scores.at[i, 'monetary']:,.0f}")
labels = {"hybrid": "Hybrid (recommended)", "collaborative": "Collaborative", "content": "Content-based", "popularity": "Popularity"}
strategy = c2.selectbox("Method", STRATEGIES, format_func=labels.get)
n = c3.slider("How many", 3, 10, 5)

df = reco.recommend(int(cid), n=n, strategy=strategy)
history = reco.data.history.get(int(cid), [])
row = scores.loc[cid]
theme.html(f'<div class="rq-note" style="margin:6px 0 14px">{theme.badge(row["segment"])} &nbsp; has bought <b>{len(history)}</b> different products so far.</div>')

if df.empty:
    st.info("No recommendations available for this customer.")
    st.stop()

left, right = st.columns([1.35, 1], gap="large")
with left:
    theme.section("Top picks", "Bar = how the score is made up. Hover a bar for details." if strategy == "hybrid" else "Ranked by the selected method.")
    cards = []
    for rank, r in enumerate(df.itertuples(), 1):
        if strategy == "hybrid":
            parts = [(WEIGHTS["collaborative"] * r.collaborative, theme.BLUE, "Collaborative"), (WEIGHTS["content"] * r.content, "#1baf7a", "Content"),
                     (WEIGHTS["popularity"] * r.popularity, "#eda100", "Popularity")]
            tot = sum(p[0] for p in parts) or 1
            meter = "".join(f'<i title="{nm} {v:.2f}" style="width:{v / tot * 100:.1f}%;background:{col}"></i>' for v, col, nm in parts)
        else:
            meter = f'<i style="width:{max(r.score, 0) * 100:.0f}%;background:{theme.INDIGO}"></i>'
        cards.append(f'<div class="rq-reco"><div class="rq-rank">{rank}</div><div><div class="t">{theme.esc(r.description.title())}</div>'
                     f'<div class="c">Code {theme.esc(r.stock_code)}</div><div class="r">{theme.esc(r.reason)}</div></div>'
                     f'<div><div class="rq-score">{r.score:.2f}</div><div class="rq-meter">{meter}</div></div></div>')
    theme.html("".join(cards))
    if strategy == "hybrid":
        theme.html(f'<div class="rq-note"><span class="rq-dot" style="background:{theme.BLUE}"></span> Collaborative {WEIGHTS["collaborative"]:.0%} &nbsp; '
                   f'<span class="rq-dot" style="background:#1baf7a"></span> Content {WEIGHTS["content"]:.0%} &nbsp; '
                   f'<span class="rq-dot" style="background:#eda100"></span> Popularity {WEIGHTS["popularity"]:.0%}</div>')
with right:
    theme.section("Score breakdown", "Each signal is scaled 0-1 before blending.")
    d = df.iloc[::-1]
    names = [x.title()[:26] for x in d["description"]]
    fig = go.Figure()
    for col, colour, w in (("collaborative", theme.BLUE, WEIGHTS["collaborative"]), ("content", "#1baf7a", WEIGHTS["content"]), ("popularity", "#eda100", WEIGHTS["popularity"])):
        fig.add_bar(y=names, x=d[col], name=col.title(), orientation="h", marker=dict(color=colour, line=dict(color="rgba(0,0,0,0)", width=0)),
                    hovertemplate="%{y}<br>" + col.title() + " score %{x:.2f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.05)
    fig.update_xaxes(range=[0, 1.02], showgrid=True, title="Normalised score")
    fig.update_yaxes(showgrid=False)
    theme.show(theme.style(fig, 90 + 62 * len(df)))

theme.section("Already in this customer's history", "A sample of what the engine learned from.")
cur = reco.data.items.loc[history, ["StockCode", "Description", "n_orders"]].sort_values("n_orders", ascending=False).head(8)
theme.show_df(cur.rename(columns={"StockCode": "Code", "Description": "Product", "n_orders": "Orders (all customers)"}))

with st.expander("How good are these recommendations? Time-aware evaluation"):
    st.caption(f"Trained on sales before {ev['split_date'][:10]}, tested on what {ev['n_eval_customers']} customers bought for the first time afterwards. "
               "A random pick would score near zero.")
    rows = [{"Method": labels[k], "Precision@5": v["precision@5"], "Recall@5": v["recall@5"], "NDCG@5": v["ndcg@5"], "Coverage": v["coverage"], "Avg. popularity percentile": v["avg_popularity_percentile"]}
            for k, v in ev["models"].items()]
    theme.show_df(pd.DataFrame(rows), column_config={
        "Precision@5": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=0.3), "Recall@5": st.column_config.NumberColumn(format="%.3f"),
        "NDCG@5": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=0.3), "Coverage": st.column_config.NumberColumn(format="percent"),
        "Avg. popularity percentile": st.column_config.NumberColumn(format="%.2f", help="1.0 = only best-sellers. Lower means more varied recommendations.")})
    st.caption("Popularity bias: the collaborative method recommends almost only best-sellers. The hybrid and content methods reach further into the catalogue (higher coverage).")
