# Before / after data-quality report

| Metric | Before | After |
|---|---:|---:|
| rows | 1067371 | 1021241 |
| columns | 8 | 8 |
| date_min | 2009-12-01 07:45:00 | 2009-12-01 07:45:00 |
| date_max | 2011-12-09 12:50:00 | 2011-12-09 12:50:00 |
| unique_invoices | 53628 | 46916 |
| unique_customers | 5942 | 5875 |
| unique_products | 5305 | 4737 |
| missing_customer_id | 243007 | 227089 |
| missing_customer_id_pct | 22.77 | 22.24 |
| missing_description | 4382 | 0 |
| duplicate_rows | 34335 | 2 |
| cancelled_invoices_rows | 19494 | 17912 |
| negative_quantity_rows | 22950 | 17912 |
| negative_qty_not_cancellation | 3457 | 0 |
| zero_quantity_rows | 0 | 0 |
| zero_or_negative_price_rows | 6207 | 0 |
| non_product_stock_code_rows | 5830 | 0 |
| max_quantity | 80995 | 9456 |
| min_quantity | -80995 | -9360 |
| max_unit_price | 38970.0 | 1157.15 |

## Cleaning steps

| Step | Rows removed | Why |
|---|---:|---|
| drop unparsable date/quantity/price | 0 | Rows whose core fields cannot be parsed are unusable. |
| remove exact duplicates | 34,335 | Identical rows are double-entries, not double purchases. |
| remove service stock codes | 5,818 | POST/Discount/Manual/Bank charges are not products. |
| remove zero quantity | 0 | A line with no units carries no information. |
| remove zero/negative price | 5,964 | Free items and stock adjustments are not sales. |
| remove data-entry outliers | 13 | |Quantity| >= 10000 or UnitPrice >= 5000; the matching reversal is removed too. |
| remove contradictory cancellations | 0 | Cancellation invoices with positive quantity. |
| remove unexplained negatives | 0 | Negative quantity without a cancellation invoice (stock write-offs). |
| drop products with no description anywhere | 0 | Cannot be used by the content-based recommender. |

Data source: **uci_online_retail_ii**
