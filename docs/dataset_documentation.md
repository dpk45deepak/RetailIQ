# Dataset documentation - Online Retail II

**Source:** UCI Machine Learning Repository, *Online Retail II* - <https://archive.ics.uci.edu/dataset/502/online+retail+ii> (licence CC BY 4.0).
**Content:** every sale line of a UK-based online gift retailer, 1 Dec 2009 - 9 Dec 2011. Many customers are wholesalers.

> **Provenance of the copy used for the committed results.** The build environment could not reach UCI, so the data was read from a GitHub-hosted copy of the same dataset (the R package `onlineretail2`). Its row count (1,067,371), column set and date range match the UCI file exactly, but it was **not compared byte-for-byte**. To use the official file run `python scripts/download_data.py` and then `python scripts/run_pipeline.py`; all numbers in this repo regenerate.

## Structure (raw)

| Quantity | Value |
|---|---|
| Rows (observations = one product line on an invoice) | 1,067,371 |
| Columns (features) | 8 |
| Unique invoices | 53,628 |
| Unique customers | 5,942 |
| Unique StockCodes | 5,304 |
| Period | 2009-12-01 07:45 to 2011-12-09 12:50 |

## Column dictionary

| Column (standardised) | UCI name (Online Retail II) | Type | Meaning | Quality notes |
|---|---|---|---|---|
| `InvoiceNo` | Invoice | text | Order number; leading **C** = cancellation | |
| `StockCode` | StockCode | text | Product code | Contains non-product codes (`POST`, `D`, `M`, `BANK CHARGES`, ...) |
| `Description` | Description | text | Product name | 4,382 missing; inconsistent case/spacing |
| `Quantity` | Quantity | int | Units on the line | Negative for returns and write-offs; +/-80,995 extremes |
| `InvoiceDate` | InvoiceDate | datetime | Order timestamp | Last month (Dec 2011) incomplete |
| `UnitPrice` | Price | float, GBP | Price per unit | 6,207 rows with price <= 0; max 38,970 (manual/fee lines) |
| `CustomerID` | Customer ID | float -> int | Customer identifier | **243,007 missing (22.8%)** = guest checkouts |
| `Country` | Country | text | Customer country | 85% of net revenue is the United Kingdom |

## Data-quality problems and decisions (raw counts from `data/processed/raw_profile.json`)

| Problem | Rows | Decision |
|---|---|---|
| Exact duplicate rows | 34,335 | Removed |
| Cancellation invoices (`C...`) | 19,494 | **Kept as `return` records**; net revenue = sales + (negative) returns; frequency counts only sale invoices |
| Negative quantity | 22,950 (3,457 not cancellations) | Cancellations kept as returns; the rest are priced 0 and removed by the price rule |
| Price <= 0 | 6,207 | Removed (free items, bad-debt adjustments) |
| Service/non-product codes | 5,830 | Removed from sales analysis (postage, fees, discounts, samples) |
| Missing CustomerID | 243,007 | Kept for company-level revenue and product statistics; excluded from customer-level features |
| Missing description | 4,382 | Filled from the canonical description of the same StockCode |
| Extreme quantities / prices | 13 lines removed; 1,002 lines flagged `IsOutlier` | Absurd data-entry blow-ups removed; large wholesale lines flagged, not removed |

After cleaning: **1,021,241 rows** (1,003,329 sale lines + 17,912 return lines). Gross sales GBP 19.38M, returns GBP 0.47M (2.43%), **net revenue GBP 18.91M**. Guest rows are 22.2% of rows and 13.6% of revenue. Full step-by-step report: [`data_quality_report.md`](data_quality_report.md).

## Customer-level features (`data/processed/customer_features.parquet`)

Snapshot date = day after the last transaction (2011-12-10), observation window = all earlier data. 5,852 customers with at least one purchase after cleaning.

`recency_days, frequency, monetary, avg_order_value, total_items, product_diversity, active_months, avg_basket_size, spend_30d, spend_60d, spend_90d, avg_days_between_purchases, tenure_days, return_rate` (+ `country`, `snapshot_date`). Definitions are in notebook 03.

## Known limitations
* One retailer, two years: seasonality (Q4 peak) cannot be separated from growth with a single Christmas per year.
* Customer IDs are pseudonymous; no demographics, so segments are behavioural only.
* Guest checkouts (13.6% of revenue) are invisible to customer models.
