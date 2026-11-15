import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_transactions, normalise_description
from src.data.load_data import standardise_columns
from src.data.validate_data import check_schema, profile_dataframe
from tests.conftest import make_raw

RAW = [
    ("1001", "A1", "  heart  mug ", 2, "2011-01-05 10:00", 3.0, 1, "United Kingdom"),     # normal sale
    ("1001", "A1", "  heart  mug ", 2, "2011-01-05 10:00", 3.0, 1, "United Kingdom"),     # exact duplicate
    ("1001", "B2", "Blue Bowl", 1, "2011-01-05 10:00", 10.0, 1, "United Kingdom"),
    ("C1002", "A1", "HEART MUG", -1, "2011-01-09 11:00", 3.0, 1, "United Kingdom"),        # genuine return
    ("1003", "POST", "POSTAGE", 1, "2011-01-10 09:00", 18.0, 2, "France"),                 # service code
    ("1004", "B2", "Blue Bowl", 1, "2011-01-11 09:00", 0.0, 2, "France"),                  # free item
    ("1005", "B2", "Blue Bowl", 0, "2011-01-11 09:00", 10.0, 2, "France"),                 # zero quantity
    ("1006", "C3", "Glitter Pen", 5, "2011-01-12 09:00", 1.0, None, "United Kingdom"),     # guest checkout
    ("1007", "D4", "Giant Rug", 80000, "2011-01-13 09:00", 2.0, 3, "United Kingdom"),      # data-entry blow-up
    ("1008", "B2", "Blue Bowl", -3, "2011-01-14 09:00", 10.0, 2, "France"),                # negative w/o cancellation
]


@pytest.fixture
def cleaned():
    raw = make_raw(RAW)
    return raw, *clean_transactions(raw)


def test_input_is_never_modified(cleaned):
    raw, _, _ = cleaned
    assert len(raw) == len(RAW) and raw["Description"].iloc[0] == "  heart  mug "


def test_each_problem_is_handled(cleaned):
    _, clean, report = cleaned
    assert clean.duplicated(["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice"]).sum() == 0
    assert not clean["StockCode"].isin(["POST"]).any()
    assert (clean["UnitPrice"] > 0).all() and (clean["Quantity"] != 0).all()
    assert clean["Quantity"].abs().max() < 10_000
    assert not ((clean["Quantity"] < 0) & ~clean["InvoiceNo"].str.startswith("C")).any()
    removed = {s["step"]: s["rows_removed"] for s in report["steps"]}
    assert removed["remove exact duplicates"] == 1 and removed["remove service stock codes"] == 1


def test_returns_are_kept_separately_and_netted(cleaned):
    _, clean, report = cleaned
    rets = clean[clean["RecordType"] == "return"]
    assert len(rets) == 1 and (rets["Revenue"] < 0).all()
    sales = clean[clean["RecordType"] == "sale"]
    assert report["summary"]["net_revenue"] == pytest.approx(sales["Revenue"].sum() + rets["Revenue"].sum())
    assert report["summary"]["returns_value"] == pytest.approx(-3.0)


def test_guest_rows_kept_but_flagged(cleaned):
    _, clean, _ = cleaned
    guest = clean[~clean["HasCustomer"]]
    assert len(guest) == 1 and guest["CustomerID"].isna().all()


def test_descriptions_normalised_and_canonical(cleaned):
    _, clean, _ = cleaned
    assert set(clean.loc[clean["StockCode"] == "A1", "Description"]) == {"HEART MUG"}
    assert normalise_description(pd.Series([" a   b "])).iloc[0] == "A B"


def test_dates_are_datetime_and_sorted(cleaned):
    _, clean, _ = cleaned
    assert pd.api.types.is_datetime64_any_dtype(clean["InvoiceDate"])
    assert clean["InvoiceDate"].is_monotonic_increasing


def test_standardise_columns_handles_both_uci_layouts():
    ii = pd.DataFrame(columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])
    v1 = pd.DataFrame(columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    assert list(standardise_columns(ii).columns) == list(standardise_columns(v1).columns)
    with pytest.raises(ValueError):
        standardise_columns(ii.drop(columns=["Country"]))


def test_check_schema_and_profile(cleaned):
    raw, _, _ = cleaned
    assert check_schema(raw) == []
    assert any("missing" in p for p in check_schema(raw.drop(columns=["Quantity"])))
    assert any("datetime" in p for p in check_schema(raw.assign(InvoiceDate=raw["InvoiceDate"].astype(str))))
    prof = profile_dataframe(raw)
    assert prof["rows"] == len(raw)


def test_cleaning_on_synthetic_data_is_idempotent_in_size(raw_synth, clean_synth):
    again, _ = clean_transactions(clean_synth.drop(columns=["Revenue", "RecordType", "HasCustomer", "IsOutlier"]))
    assert len(again) == len(clean_synth)
    assert len(clean_synth) < len(raw_synth)
