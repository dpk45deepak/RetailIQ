"""Shared fixtures. Heavy fixtures are session-scoped so the whole suite stays fast."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import clean_transactions  # noqa: E402
from src.data.load_data import standardise_columns  # noqa: E402
from src.data.synthetic import generate_synthetic_retail  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


def make_raw(rows: list[tuple]) -> pd.DataFrame:
    """rows: (InvoiceNo, StockCode, Description, Quantity, 'YYYY-MM-DD HH:MM', UnitPrice, CustomerID|None, Country)."""
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype("float64")
    return df


@pytest.fixture(scope="session")
def raw_synth() -> pd.DataFrame:
    return standardise_columns(generate_synthetic_retail(n_customers=300, n_products=250, seed=7))


@pytest.fixture(scope="session")
def clean_synth(raw_synth) -> pd.DataFrame:
    return clean_transactions(raw_synth)[0]


@pytest.fixture(scope="session")
def pipeline_root(tmp_path_factory) -> Path:
    """Run the FULL pipeline once on synthetic data in a temp folder (integration fixture)."""
    root = tmp_path_factory.mktemp("retailiq_root")
    run_pipeline(root, fast=True, n_customers=300, track=False)
    return root
