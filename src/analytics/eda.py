"""Exploratory-data-analysis helpers shared by notebook 02 and the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sales_only(clean: pd.DataFrame) -> pd.DataFrame:
    return clean[clean["RecordType"] == "sale"]


def kpis(clean: pd.DataFrame) -> dict:
    """Headline numbers. Revenue is NET (sales minus returns); AOV uses gross sales per sale invoice."""
    s, r = sales_only(clean), clean[clean["RecordType"] == "return"]
    n_orders = s["InvoiceNo"].nunique()
    return {
        "transactions": int(n_orders), "line_items": int(len(s)),
        "customers": int(s.loc[s["HasCustomer"], "CustomerID"].nunique()), "products": int(s["StockCode"].nunique()),
        "net_revenue": float(clean["Revenue"].sum()), "gross_sales": float(s["Revenue"].sum()),
        "returns_value": float(-r["Revenue"].sum()), "avg_order_value": float(s["Revenue"].sum() / n_orders),
        "return_rate": float(-r["Revenue"].sum() / s["Revenue"].sum()),
        "countries": int(clean["Country"].nunique()),
        "date_min": clean["InvoiceDate"].min(), "date_max": clean["InvoiceDate"].max(),
    }


def monthly_revenue(clean: pd.DataFrame) -> pd.DataFrame:
    m = clean.set_index("InvoiceDate")["Revenue"].resample("MS").sum().rename("net_revenue").reset_index()
    m["orders"] = sales_only(clean).set_index("InvoiceDate")["InvoiceNo"].resample("MS").nunique().to_numpy()
    return m.rename(columns={"InvoiceDate": "month"})


def country_revenue(clean: pd.DataFrame) -> pd.DataFrame:
    g = clean.groupby("Country")["Revenue"].sum().sort_values(ascending=False).rename("net_revenue").reset_index()
    g["share"] = g["net_revenue"] / g["net_revenue"].sum()
    return g


def weekday_hour_pattern(clean: pd.DataFrame) -> pd.DataFrame:
    s = sales_only(clean)
    inv = s.groupby("InvoiceNo").agg(date=("InvoiceDate", "min"), revenue=("Revenue", "sum")).reset_index()
    inv["weekday"] = inv["date"].dt.day_name()
    inv["hour"] = inv["date"].dt.hour
    return inv


def top_products(clean: pd.DataFrame, n: int = 10, by: str = "revenue") -> pd.DataFrame:
    s = sales_only(clean)
    g = s.groupby(["StockCode", "Description"]).agg(units=("Quantity", "sum"), revenue=("Revenue", "sum"), orders=("InvoiceNo", "nunique")).reset_index()
    return g.sort_values({"revenue": "revenue", "units": "units"}.get(by, "revenue"), ascending=False).head(n)


def customer_order_counts(clean: pd.DataFrame) -> pd.Series:
    s = sales_only(clean)
    return s[s["HasCustomer"]].groupby("CustomerID")["InvoiceNo"].nunique().rename("orders")


def distribution_stats(x: pd.Series) -> dict:
    """Mean / median / std / skewness - the vocabulary of 'what does this distribution look like?'."""
    x = x.dropna()
    return {"mean": float(x.mean()), "median": float(x.median()), "std": float(x.std()), "skew": float(x.skew()),
            "min": float(x.min()), "p95": float(x.quantile(0.95)), "max": float(x.max())}


def correlation_frame(clean: pd.DataFrame) -> pd.DataFrame:
    s = sales_only(clean)
    return s[["Quantity", "UnitPrice", "Revenue"]].assign(
        Hour=s["InvoiceDate"].dt.hour, Month=s["InvoiceDate"].dt.month).corr(method="spearman")
