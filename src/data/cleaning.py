"""Data cleaning with a documented treatment for every problem.

KEY DESIGN DECISION - returns are NOT simply dropped
----------------------------------------------------
Negative quantities are real business events: customers send goods back.
If we deleted them, revenue and customer spend would be overstated.
So the cleaned table keeps two record types:

* ``sale``   - Quantity > 0, UnitPrice > 0
* ``return`` - cancellation invoices (InvoiceNo starts with "C") with Quantity < 0

``Revenue = Quantity * UnitPrice`` is therefore positive for sales and negative for
returns, and **net revenue = sum(Revenue)**. Frequency counts only sale invoices.

Removed as invalid (and counted in the report): exact duplicates, service codes
(POST, D, M, ...), zero quantity, zero/negative price, and data-entry blow-ups
(quantity >= 10,000 or unit price >= 5,000 together with their reversals).

Rows with a missing CustomerID are KEPT for company-level revenue/product analysis,
but excluded from every customer-level step (we cannot link them to a person).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.validate_data import NON_PRODUCT_CODES, profile_dataframe, quality_report_markdown
from src.utils.config import ProjectPaths, get_logger, get_paths

log = get_logger(__name__)
MAX_PLAUSIBLE_QTY = 10_000
MAX_PLAUSIBLE_PRICE = 5_000


def normalise_description(s: pd.Series) -> pd.Series:
    """Upper-case, trim and collapse whitespace so 'Heart  mug ' == 'HEART MUG'."""
    return (s.astype("string").str.upper().str.replace(r"\s+", " ", regex=True).str.strip()).astype("object")


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean raw transactions. Returns ``(clean_df, report)``; the input is never modified."""
    df = raw.copy()
    steps: list[dict] = []
    before = profile_dataframe(raw)

    def record(name: str, n_before: int, note: str) -> None:
        steps.append({"step": name, "rows_removed": int(n_before - len(df)), "note": note})

    # 1. Types ---------------------------------------------------------------------------
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    n = len(df); df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
    record("drop unparsable date/quantity/price", n, "Rows whose core fields cannot be parsed are unusable.")

    # 2. Duplicates ----------------------------------------------------------------------
    n = len(df); df = df.drop_duplicates()
    record("remove exact duplicates", n, "Identical rows are double-entries, not double purchases.")

    # 3. Non-product codes ---------------------------------------------------------------
    n = len(df)
    df = df[~(df["StockCode"].isin(NON_PRODUCT_CODES) | df["StockCode"].str.startswith("GIFT_"))]
    record("remove service stock codes", n, "POST/Discount/Manual/Bank charges are not products.")

    # 4. Invalid quantity & price --------------------------------------------------------
    n = len(df); df = df[df["Quantity"] != 0]
    record("remove zero quantity", n, "A line with no units carries no information.")
    n = len(df); df = df[df["UnitPrice"] > 0]
    record("remove zero/negative price", n, "Free items and stock adjustments are not sales.")

    # 5. Data-entry outliers (and the reversal that corrects them) ------------------------
    n = len(df)
    df = df[(df["Quantity"].abs() < MAX_PLAUSIBLE_QTY) & (df["UnitPrice"] < MAX_PLAUSIBLE_PRICE)]
    record("remove data-entry outliers", n, f"|Quantity| >= {MAX_PLAUSIBLE_QTY} or UnitPrice >= {MAX_PLAUSIBLE_PRICE}; the matching reversal is removed too.")

    # 6. Returns handled separately ---------------------------------------------------------
    is_cancel = df["InvoiceNo"].str.upper().str.startswith("C")
    df["RecordType"] = np.where((df["Quantity"] < 0) | is_cancel, "return", "sale")
    odd = (df["Quantity"] > 0) & is_cancel          # positive qty on a C-invoice: contradictory
    n = len(df); df = df[~odd.reindex(df.index, fill_value=False)]
    record("remove contradictory cancellations", n, "Cancellation invoices with positive quantity.")
    n = len(df); df = df[~((df["RecordType"] == "return") & ~df["InvoiceNo"].str.upper().str.startswith("C"))]
    record("remove unexplained negatives", n, "Negative quantity without a cancellation invoice (stock write-offs).")

    # 7. Descriptions -----------------------------------------------------------------------
    df["Description"] = normalise_description(df["Description"])
    mode_desc = (df.dropna(subset=["Description"]).groupby("StockCode")["Description"]
                   .agg(lambda s: s.value_counts().index[0]))
    df["Description"] = df["StockCode"].map(mode_desc)         # one canonical name per product
    n = len(df); df = df.dropna(subset=["Description"])
    record("drop products with no description anywhere", n, "Cannot be used by the content-based recommender.")

    # 8. Derived columns --------------------------------------------------------------------
    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["CustomerID"] = pd.array(df["CustomerID"].round(), dtype="Int64")
    df["HasCustomer"] = df["CustomerID"].notna()
    sales_rev = df.loc[df["RecordType"] == "sale", "Revenue"]
    df["IsOutlier"] = (df["RecordType"] == "sale") & (df["Revenue"] > sales_rev.quantile(0.999))
    df = df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)

    after = profile_dataframe(df.drop(columns=["Revenue", "RecordType", "HasCustomer", "IsOutlier"]))
    sales, rets = df[df.RecordType == "sale"], df[df.RecordType == "return"]
    summary = {
        "gross_sales": round(float(sales["Revenue"].sum()), 2),
        "returns_value": round(float(rets["Revenue"].sum()), 2),
        "net_revenue": round(float(df["Revenue"].sum()), 2),
        "return_rate_pct": round(float(-rets["Revenue"].sum() / sales["Revenue"].sum() * 100), 2),
        "sale_rows": int(len(sales)), "return_rows": int(len(rets)),
        "guest_row_pct": round(float((~df["HasCustomer"]).mean() * 100), 2),
        "guest_revenue_pct": round(float(df.loc[~df["HasCustomer"], "Revenue"].sum() / df["Revenue"].sum() * 100), 2),
        "flagged_outlier_rows": int(df["IsOutlier"].sum()),
        "rows_before": int(len(raw)), "rows_after": int(len(df)),
    }
    log.info("Cleaning: %s -> %s rows", f"{len(raw):,}", f"{len(df):,}")
    return df, {"before": before, "after": after, "steps": steps, "summary": summary}


def run_cleaning(raw: pd.DataFrame, paths: ProjectPaths | None = None, source: str = "unknown") -> tuple[pd.DataFrame, dict]:
    """Clean, validate and persist (parquet + JSON + markdown report)."""
    paths = (paths or get_paths()).ensure()
    clean, report = clean_transactions(raw)
    report["data_source"] = source
    assert clean["Revenue"].notna().all() and (clean.loc[clean.RecordType == "sale", "Quantity"] > 0).all()
    clean.to_parquet(paths.processed / "cleaned_transactions.parquet", index=False)
    (paths.processed / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=str))
    md = quality_report_markdown(report["before"], report["after"], "Before / after data-quality report")
    steps = "\n## Cleaning steps\n\n| Step | Rows removed | Why |\n|---|---:|---|\n" + "\n".join(
        f"| {s['step']} | {s['rows_removed']:,} | {s['note']} |" for s in report["steps"])
    (paths.docs / "data_quality_report.md").write_text(md + steps + f"\n\nData source: **{source}**\n")
    return clean, report
