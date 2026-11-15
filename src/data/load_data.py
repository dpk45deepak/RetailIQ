"""Load the raw retail transactions.

Search order inside ``data/raw/``:
  1. online_retail_II.xlsx   (UCI Online Retail II, two sheets)
  2. online_retail_II.csv
  3. Online Retail.xlsx      (older UCI Online Retail, the documented alternative)
  4. synthetic_online_retail_II.csv (generated demo data, clearly flagged as synthetic)

Column names are standardised so the rest of the code never cares which file was used:
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.synthetic import generate_synthetic_retail
from src.utils.config import ProjectPaths, get_logger, get_paths

log = get_logger(__name__)

DATASET_URL = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
COLUMN_MAP = {
    "Invoice": "InvoiceNo", "InvoiceNo": "InvoiceNo",
    "Price": "UnitPrice", "UnitPrice": "UnitPrice",
    "Customer ID": "CustomerID", "CustomerID": "CustomerID",
}
STANDARD_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)
    missing = set(STANDARD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df[STANDARD_COLUMNS].copy()


def load_raw_transactions(paths: ProjectPaths | None = None, allow_synthetic: bool = True,
                          synthetic_customers: int = 2000) -> tuple[pd.DataFrame, str]:
    """Return ``(dataframe, source)`` where source is 'uci_online_retail_ii', 'uci_online_retail' or 'synthetic'."""
    paths = (paths or get_paths()).ensure()
    xlsx2, csv2 = paths.raw / "online_retail_II.xlsx", paths.raw / "online_retail_II.csv"
    xlsx1, synth = paths.raw / "Online Retail.xlsx", paths.raw / "synthetic_online_retail_II.csv"

    if xlsx2.exists():
        log.info("Reading %s (both sheets)", xlsx2.name)
        sheets = pd.read_excel(xlsx2, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
        return standardise_columns(pd.concat(sheets.values(), ignore_index=True)), "uci_online_retail_ii"
    if csv2.exists():
        kwargs = dict(dtype={"Invoice": str, "StockCode": str}, parse_dates=["InvoiceDate"])
        try:
            df = pd.read_csv(csv2, encoding="utf-8", **kwargs)
        except UnicodeDecodeError:  # older exports of the UCI file are latin-1
            df = pd.read_csv(csv2, encoding="latin-1", **kwargs)
        return standardise_columns(df), "uci_online_retail_ii"
    if xlsx1.exists():
        df = pd.read_excel(xlsx1, dtype={"InvoiceNo": str, "StockCode": str})
        return standardise_columns(df), "uci_online_retail"
    if not allow_synthetic:
        raise FileNotFoundError(f"No dataset in {paths.raw}. Download it from {DATASET_URL}")

    if not synth.exists():
        log.warning("Real dataset not found - generating SYNTHETIC demo data (see README).")
        generate_synthetic_retail(n_customers=synthetic_customers).to_csv(synth, index=False)
    df = pd.read_csv(synth, dtype={"Invoice": str, "StockCode": str}, parse_dates=["InvoiceDate"])
    return standardise_columns(df), "synthetic"
