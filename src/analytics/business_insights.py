"""Turn model outputs and dataset facts into concrete business recommendations.

RULE: every sentence is generated from computed numbers. Nothing here is hard-coded as a "finding".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.analytics.eda import country_revenue, kpis, monthly_revenue
from src.models.customer_segmentation import SEGMENT_PLAYBOOK


def gbp(x: float) -> str:
    return f"£{x:,.0f}" if abs(x) < 1e6 else f"£{x / 1e6:,.2f}M"


def product_affinity(clean: pd.DataFrame, top_products: int = 300, min_orders: int = 25, top_n: int = 40) -> pd.DataFrame:
    """Market-basket pairs: support, confidence and lift.

    lift = P(A and B) / (P(A) * P(B)).  Lift > 1 means the products are bought together MORE often than chance.
    Restricted to the ``top_products`` best-sellers so the pair counts are statistically meaningful.
    """
    s = clean[clean["RecordType"] == "sale"]
    pop = s.groupby("StockCode")["InvoiceNo"].nunique().sort_values(ascending=False).head(top_products)
    s = s[s["StockCode"].isin(pop.index)]
    inv_codes, inv_idx = pd.factorize(s["InvoiceNo"]); item_codes, item_idx = pd.factorize(s["StockCode"])
    b = sparse.csr_matrix((np.ones(len(s)), (inv_codes, item_codes)), shape=(len(inv_idx), len(item_idx)))
    b.data[:] = 1
    co = (b.T @ b).tocoo()
    n_inv = b.shape[0]
    count = np.asarray(b.sum(axis=0)).ravel()
    desc = s.drop_duplicates("StockCode").set_index("StockCode")["Description"]
    rows = []
    for i, j, c in zip(co.row, co.col, co.data):
        if i < j and c >= min_orders:
            lift = (c / n_inv) / ((count[i] / n_inv) * (count[j] / n_inv))
            rows.append({"product_a": item_idx[i], "product_b": item_idx[j], "description_a": desc[item_idx[i]],
                         "description_b": desc[item_idx[j]], "orders_together": int(c), "support": c / n_inv,
                         "confidence_a_to_b": c / count[i], "confidence_b_to_a": c / count[j], "lift": lift})
    return pd.DataFrame(rows).sort_values("lift", ascending=False).head(top_n).reset_index(drop=True)


def retention_opportunities(scored: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """At-risk / hibernating customers ranked by historical spend (what we stand to lose or win back)."""
    pool = scored[scored["segment"].isin(["At-Risk Customers", "Hibernating Customers"])]
    cols = ["segment", "recency_days", "frequency", "monetary", "avg_order_value", "p_high_value", "country"]
    return pool.sort_values("monetary", ascending=False)[cols].head(n)


def generate_insights(clean: pd.DataFrame, scored: pd.DataFrame, profile: pd.DataFrame, importance: pd.DataFrame | None,
                      affinity: pd.DataFrame, reco_eval: dict | None) -> list[dict]:
    """Return a list of {id, title, finding, action, kind} dicts, each derived from the data."""
    out: list[dict] = []
    add = lambda id_, kind, title, finding, action: out.append(dict(id=id_, kind=kind, title=title, finding=finding, action=action))
    k = kpis(clean)

    # 1. Revenue concentration
    top = profile.iloc[0]; name = profile.index[0]
    add("concentration", "opportunity", f"{name} carry the business",
        f"{name} are {top.customer_share:.0%} of customers but {top.revenue_share:.0%} of historical net revenue ({gbp(top.revenue)}).",
        SEGMENT_PLAYBOOK.get(name, {}).get("action", "Protect this group."))

    # 2. At-risk revenue
    if "At-Risk Customers" in profile.index:
        r = profile.loc["At-Risk Customers"]
        add("at_risk", "risk", "High-value customers are going quiet",
            f"{int(r.customers):,} at-risk customers spent {gbp(r.revenue)} historically ({r.revenue_share:.0%} of revenue) but have not bought for a median of {r.recency_median:.0f} days.",
            SEGMENT_PLAYBOOK["At-Risk Customers"]["action"])

    # 3. Seasonality
    m = monthly_revenue(clean).assign(moy=lambda d: d["month"].dt.month)
    full = m.iloc[:-1] if m["month"].iloc[-1].day == 1 and len(m) > 12 else m
    by_moy = full.groupby("moy")["net_revenue"].mean()
    q4 = by_moy.loc[[x for x in (10, 11, 12) if x in by_moy.index]].sum() / by_moy.sum()
    peak = int(by_moy.idxmax()); month_name = pd.Timestamp(2000, peak, 1).strftime("%B")
    add("seasonality", "opportunity", f"{month_name} is the peak month",
        f"{month_name} averages {by_moy.max() / by_moy.mean():.1f}x the typical month, and October-December bring {q4:.0%} of annual revenue.",
        "Build stock and campaign calendars backwards from this peak; launch loyalty perks before September.")

    # 4. What drives value
    if importance is not None and len(importance):
        feats = ", ".join(importance["feature"].head(3).str.replace("_", " "))
        add("drivers", "info", "What predicts a high-value customer",
            f"The model relies most on: {feats}. (Ranked by held-out feature importance.)",
            "Use these signals to build a simple early-warning score for the CRM team.")

    # 5. Product affinity
    if len(affinity):
        a = affinity.iloc[0]
        add("affinity", "opportunity", "Products that sell together",
            f"{a.description_a.title()} and {a.description_b.title()} appear together in {a.orders_together:,} orders (lift {a.lift:.1f}x; "
            f"{a.confidence_a_to_b:.0%} of orders with the first also contain the second).",
            "Bundle them, or show one on the other's product page and basket.")

    # 6. Returns
    add("returns", "risk" if k["return_rate"] > 0.05 else "info", "Returns are worth watching",
        f"Returns equal {k['return_rate']:.1%} of gross sales ({gbp(k['returns_value'])}).",
        "Track return rate by product and customer; investigate any item well above the average.")

    # 7. Geography
    cr = country_revenue(clean)
    add("geography", "opportunity", "Revenue is concentrated in one market",
        f"{cr.iloc[0].Country} generates {cr.iloc[0].share:.0%} of net revenue; the next market, {cr.iloc[1].Country}, only {cr.iloc[1].share:.1%}.",
        "Test export-focused promotions in the 2-3 largest secondary markets before broad expansion.")

    # 8. One-time buyers
    one = (scored["frequency"] == 1).mean()
    add("one_time", "risk", "Many customers never come back",
        f"{one:.0%} of customers have placed exactly one order.",
        "Add a second-purchase journey: replenishment reminder and affinity-based recommendation within 30 days.")

    # 9. Recommender lift
    if reco_eval:
        mm = reco_eval["models"]; key = [c for c in mm["hybrid"] if c.startswith("precision@")][0]
        h, p = mm["hybrid"][key], mm["popularity"][key]
        if p > 0:
            add("reco", "opportunity", "Personalised recommendations beat best-sellers",
                f"In a time-aware test the hybrid recommender reached {key} {h:.3f} versus {p:.3f} for a popularity baseline ({h / p - 1:+.0%}).",
                "Use the hybrid list in post-purchase emails and on the basket page; A/B test against best-sellers.")

    # 10. Guest checkouts
    guest = clean.loc[~clean["HasCustomer"], "Revenue"].sum() / clean["Revenue"].sum()
    add("guests", "info", "Some revenue cannot be linked to a customer",
        f"Guest checkouts without a CustomerID account for {guest:.0%} of net revenue and are invisible to segmentation and recommendations.",
        "Prompt guests to create an account at checkout: it improves segmentation and recommendations.")
    return out
