"""Data-quality inspection (used before AND after cleaning).

``profile_dataframe`` answers: how big is it, what types, how much is missing,
and which known retail-specific problems are present?
"""
from __future__ import annotations

import pandas as pd

REQUIRED = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
NON_PRODUCT_CODES = {"POST", "D", "M", "DOT", "BANK CHARGES", "CRUK", "AMAZONFEE", "ADJUST", "ADJUST2", "TEST001", "TEST002", "S", "B", "C2", "PADS"}

COLUMN_GUIDE = {
    "InvoiceNo": "Invoice id (6 digits). A leading 'C' means the invoice is a cancellation.",
    "StockCode": "Product code. A few codes are services (POST, D, M, BANK CHARGES), not products.",
    "Description": "Product name (free text, sometimes missing or inconsistent).",
    "Quantity": "Units per line. Negative = returned/cancelled.",
    "InvoiceDate": "Date and time of the invoice.",
    "UnitPrice": "Price per unit in GBP.",
    "CustomerID": "5-digit customer id. Missing for guest checkouts.",
    "Country": "Customer's country.",
}


def check_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with the column layout (empty list = OK)."""
    problems = [f"missing column: {c}" for c in REQUIRED if c not in df.columns]
    if not problems and not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        problems.append("InvoiceDate is not datetime")
    return problems


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Compact data-quality report, JSON-serialisable."""
    inv = df["InvoiceNo"].astype(str)
    qty, price = df["Quantity"], df["UnitPrice"]
    stock = df["StockCode"].astype(str).str.upper()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "date_min": str(pd.to_datetime(df["InvoiceDate"]).min()),
        "date_max": str(pd.to_datetime(df["InvoiceDate"]).max()),
        "unique_invoices": int(inv.nunique()),
        "unique_customers": int(df["CustomerID"].nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "missing_customer_id": int(df["CustomerID"].isna().sum()),
        "missing_customer_id_pct": round(float(df["CustomerID"].isna().mean() * 100), 2),
        "missing_description": int(df["Description"].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "cancelled_invoices_rows": int(inv.str.startswith("C").sum()),
        "negative_quantity_rows": int((qty < 0).sum()),
        "negative_qty_not_cancellation": int(((qty < 0) & ~inv.str.startswith("C")).sum()),
        "zero_quantity_rows": int((qty == 0).sum()),
        "zero_or_negative_price_rows": int((price <= 0).sum()),
        "non_product_stock_code_rows": int(stock.isin(NON_PRODUCT_CODES).sum()),
        "max_quantity": int(qty.max()),
        "min_quantity": int(qty.min()),
        "max_unit_price": float(price.max()),
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items()},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


def quality_report_markdown(before: dict, after: dict | None = None, title: str = "Data-quality report") -> str:
    """Render one or two profiles as a markdown table."""
    keys = [k for k in before if not isinstance(before[k], dict)]
    lines = [f"# {title}", "", "| Metric | Before |" + (" After |" if after else ""), "|---|---:|" + ("---:|" if after else "")]
    for k in keys:
        row = f"| {k} | {before[k]} |" + (f" {after.get(k, '')} |" if after else "")
        lines.append(row)
    return "\n".join(lines) + "\n"
