"""Generate the six teaching notebooks (and optionally execute them so outputs are stored).

    python scripts/build_notebooks.py            # build + execute
    python scripts/build_notebooks.py --no-exec  # build only

Notebooks are for EXPERIMENTATION and EXPLANATION; all reusable logic lives in ``src/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = '''import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))                      # lets us import our own code from src/
from src.utils.config import get_paths, RANDOM_STATE
paths = get_paths(ROOT)

sns.set_theme(style="whitegrid", context="notebook", palette=["#2563eb", "#0d9488", "#4f46e5", "#eda100", "#eb6834", "#e87ba4"])
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "figure.figsize": (9, 4.2)})
pd.options.display.float_format = "{:,.2f}".format
pd.options.display.width = 140'''


def briefing(title: str, what: str, why: str, problem: str, code: str, output: str, wrong: str, pros: str) -> str:
    return f"""# {title}

> **Mentor briefing** - the seven questions we answer for every phase.

| Question | Answer |
|---|---|
| **1. What are we doing?** | {what} |
| **2. Why?** | {why} |
| **3. Real-world problem it solves** | {problem} |
| **4. What does the code do?** | {code} |
| **5. Expected output** | {output} |
| **6. What can go wrong?** | {wrong} |
| **7. How do professionals approach this?** | {pros} |
"""


def build(name: str, cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    nbf.write(nb, NB_DIR / name)
    return nb


md = lambda s: ("md", s)
py = lambda s: ("py", s)

# =====================================================================================================
NB1 = [
    md(briefing("Notebook 1 - Data understanding",
        "Load the Online Retail II file, look at it from every angle and write down what is wrong with it.",
        "A model can only be as good as its data. Ten minutes of inspection saves days of debugging a model that learned from garbage.",
        "A retailer exports every sale line to a spreadsheet. Before anyone can report revenue or build recommendations, someone must know what each column means and which rows can be trusted.",
        "`load_raw_transactions` reads the file and standardises column names; `profile_dataframe` and a few pandas one-liners measure rows, types, missing values and data-quality problems.",
        "Roughly one million rows x 8 columns, about a quarter of them without a CustomerID, plus cancellations, free items, service codes and duplicates.",
        "Excel stores dates and codes inconsistently (a StockCode like `85123A` can be turned into a number); wrong encodings; two sheets that must be combined; silently treating guest checkouts as one huge customer.",
        "They profile first, decide cleaning rules second, and write every decision down (see `docs/dataset_documentation.md`). They never clean by guessing.")),
    md("""## Concepts you need first

* **Dataset** - a table of recorded facts. Here: one row per *product line on an invoice*.
* **Observation** (a row, also called a record or sample) - one thing we measured. Here: "invoice 489434 contained 12 units of product 85048".
* **Feature** (a column, also called a variable) - one property of every observation: `Quantity`, `UnitPrice`, `Country` ...
* **EDA (Exploratory Data Analysis)** - looking at the data with summaries and charts *before* modelling, to learn its shape, spot problems and form questions.
* **Missing data** - cells with no value (`NaN`). Missing is a *message*: a missing CustomerID here means a guest checkout, not a typo.
* **Why inspect before ML?** Models do not know the data is broken. They will happily learn that a 80,995-unit order is normal.

### Where the data comes from
UCI Machine Learning Repository, *Online Retail II*: transactions of a UK-based online gift shop, **1 Dec 2009 - 9 Dec 2011**. Official page: <https://archive.ics.uci.edu/dataset/502/online+retail+ii>."""),
    py(SETUP),
    md("## 1. Load the file"),
    py('''from src.data.load_data import load_raw_transactions, DATASET_URL
raw, source = load_raw_transactions(paths, allow_synthetic=False)
print("data source :", source)
print("shape       :", raw.shape, "(rows, columns)")
raw.head()'''),
    md("""Our loader accepts the official `online_retail_II.xlsx` (two sheets), a CSV export, or the older *Online Retail* file, and renames columns so everything downstream sees the same eight names."""),
    py("raw.dtypes.to_frame('dtype').T"),
    md("""### Column dictionary

| Column | Meaning | Type | Notes |
|---|---|---|---|
| `InvoiceNo` | Order number | text | A leading **C** means *cancellation / return* |
| `StockCode` | Product code | text | Some codes are not products (`POST`, `M`, `D`, `BANK CHARGES` ...) |
| `Description` | Product name | text | Inconsistent spelling/case; sometimes missing |
| `Quantity` | Units on this line | integer | **Negative** for returns and stock adjustments |
| `InvoiceDate` | When the order was placed | datetime | Used for every time-based step |
| `UnitPrice` | Price per unit, GBP | float | Zero for free items / adjustments |
| `CustomerID` | Customer identifier | float -> integer | **Missing** for guest checkouts |
| `Country` | Customer country | text | 40+ countries, mostly United Kingdom |"""),
    md("## 2. Missing values"),
    py('''missing = pd.DataFrame({"missing": raw.isna().sum(), "percent": raw.isna().mean() * 100}).sort_values("missing", ascending=False)
missing'''),
    md("""`CustomerID` is the big one. We **cannot** build customer-level features for those rows - but they are still real sales, so we keep them for company-level revenue and product analysis (see Notebook 3)."""),
    py('''from src.data.validate_data import check_schema, profile_dataframe, NON_PRODUCT_CODES
assert check_schema(raw) == [], check_schema(raw)       # fails loudly if a column is missing or InvoiceDate is not a datetime
print("schema OK")
raw.describe(include="all").T[["count", "unique", "mean", "min", "50%", "max"]]'''),
    md("""Look at `Quantity` and `UnitPrice`: the **minimum is negative** and the **maximum is enormous**. The mean and the median are far apart. That is the first sign of a *skewed* distribution with outliers (Notebook 2)."""),
    md("## 3. Hunt for data-quality problems\nEach line below measures one suspicion."),
    py('''inv = raw["InvoiceNo"].astype(str)
is_cancel = inv.str.upper().str.startswith("C")
problems = pd.Series({
    "exact duplicate rows": raw.duplicated().sum(),
    "cancellation invoices (InvoiceNo starts with C)": is_cancel.sum(),
    "negative quantity": (raw["Quantity"] < 0).sum(),
    "negative quantity WITHOUT a cancellation invoice": ((raw["Quantity"] < 0) & ~is_cancel).sum(),
    "zero quantity": (raw["Quantity"] == 0).sum(),
    "unit price == 0": (raw["UnitPrice"] == 0).sum(),
    "unit price < 0": (raw["UnitPrice"] < 0).sum(),
    "missing CustomerID (guest checkout)": raw["CustomerID"].isna().sum(),
    "missing Description": raw["Description"].isna().sum(),
    "service / non-product StockCode": raw["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES).sum(),
    "|Quantity| >= 10,000 (suspicious)": (raw["Quantity"].abs() >= 10_000).sum(),
}).rename("rows").to_frame()
problems["percent of all rows"] = problems["rows"] / len(raw) * 100
problems'''),
    py('''# Look at the evidence instead of trusting the counts.
print("A normal cancellation - customer returns goods:")
display(raw[is_cancel].head(3))
print("The same product being ordered AND cancelled in a huge quantity (a data-entry error and its correction):")
display(raw[raw["Quantity"].abs() >= 10_000].sort_values("InvoiceDate"))'''),
    md("""### What this tells us (decisions for Notebook 3 / `src/data/cleaning.py`)

| Finding | Decision |
|---|---|
| Duplicate rows | Remove - identical rows are double entries |
| Cancellations (`C...`, negative quantity) | **Keep as `return` records**, do not delete. Net revenue = sales - returns |
| Negative quantity without a `C` invoice | Remove - stock write-offs, not customer behaviour |
| Zero / negative price | Remove - free items and accounting adjustments are not sales |
| Service codes (`POST`, `D`, `M` ...) | Remove from product analysis - they are fees, not products |
| Quantity >= 10,000 or price >= 5,000 | Remove with their reversal - data-entry blow-ups that would distort every average |
| Missing CustomerID | Keep for company analysis, exclude from customer analysis |
| Messy descriptions | Upper-case, trim, one canonical name per StockCode |"""),
    py('''print("Orders (unique invoices):", raw["InvoiceNo"].nunique())
print("Customers (known)       :", int(raw["CustomerID"].nunique()))
print("Products (StockCodes)   :", raw["StockCode"].nunique())
print("Countries               :", raw["Country"].nunique())
print("Period                  :", raw["InvoiceDate"].min(), "->", raw["InvoiceDate"].max())'''),
    md("""## Check yourself
1. Why is *deleting every negative quantity* a mistake? (Hint: what happens to revenue?)
2. A customer appears with `CustomerID = NaN` in 22% of rows. Why can't we simply treat them as "customer NaN"?
3. Which columns would you expect to be **skewed**? Verify it in Notebook 2.

**Acceptance for Phase 1:** the dataset loads, its structure is documented (`docs/dataset_documentation.md`) and every data-quality problem has a named treatment. Tick."""),
]

# =====================================================================================================
NB2 = [
    md(briefing("Notebook 2 - Exploratory data analysis",
        "Answer the basic business questions (how much, who, what, when, where) and learn to read distributions.",
        "EDA tells you which features matter, which transformations you will need (logs, scaling) and which business stories the data can support.",
        "A manager asks: 'How big are we? Where does the money come from? Do a few customers dominate?' Charts answer faster than tables.",
        "Loads the cleaned table and computes KPIs, monthly revenue, country revenue, top products, order counts and distributions.",
        "Seven charts plus numbers: revenue peaks in autumn, the UK dominates, spend per customer is extremely right-skewed.",
        "Reading averages of skewed data; forgetting returns when summing revenue; mistaking a partial last month for a revenue crash; ignoring guest checkouts.",
        "They chart every important column, quantify what they see (median, skew, share) and always ask 'so what for the business?'.")),
    md("""## Statistics vocabulary (read this once)

* **Mean** - the average. Pulled hard by extreme values.
* **Median** - the middle value. 50% are below, 50% above. Robust to outliers.
* **Standard deviation (std)** - typical distance from the mean. Big std = values are spread out.
* **Skewness** - lopsidedness. 0 = symmetric; **> 1 = long right tail** (a few huge values). Money and counts are almost always right-skewed.
* **Correlation** - how two variables move together, from -1 to +1. *Not* causation. We use **Spearman** (rank-based) because it is not fooled by outliers.
* **Histogram** - bars showing how many observations fall in each value range. **Box plot** - the box is the middle 50%, the line the median, dots are outliers."""),
    py(SETUP + '''
from src.analytics import eda
clean = pd.read_parquet(paths.processed / "cleaned_transactions.parquet")
sales = eda.sales_only(clean)
print(f"{len(clean):,} cleaned lines  |  {len(sales):,} sale lines  |  {(clean.RecordType=='return').sum():,} return lines")'''),
    md("## 1. Headline numbers"),
    py('''k = eda.kpis(clean)
pd.Series({
    "Orders (sale invoices)": f"{k['transactions']:,}", "Sale lines": f"{k['line_items']:,}",
    "Unique customers": f"{k['customers']:,}", "Unique products": f"{k['products']:,}",
    "Gross sales": f"GBP {k['gross_sales']:,.0f}", "Returns": f"GBP {k['returns_value']:,.0f} ({k['return_rate']:.1%})",
    "NET revenue": f"GBP {k['net_revenue']:,.0f}", "Average order value": f"GBP {k['avg_order_value']:,.2f}",
    "Countries": k["countries"], "Period": f"{k['date_min']:%d %b %Y} - {k['date_max']:%d %b %Y}"}).to_frame("value")'''),
    md("""*Net revenue = sales + (negative) returns.* This is why Notebook 1 insisted on keeping returns."""),
    md("## 2. Monthly revenue trend (line chart)"),
    py('''m = eda.monthly_revenue(clean)
fig, ax = plt.subplots()
ax.plot(m["month"], m["net_revenue"] / 1e6, color="#2563eb", lw=2.5, marker="o", ms=4)
ax.fill_between(m["month"], m["net_revenue"] / 1e6, alpha=.12, color="#2563eb")
peak = m.loc[m["net_revenue"].idxmax()]
ax.annotate(f"peak {peak['month']:%b %Y}\\nGBP {peak['net_revenue']/1e6:.2f}M", (peak["month"], peak["net_revenue"] / 1e6), xytext=(-90, -5), textcoords="offset points", arrowprops=dict(arrowstyle="-", color="grey"))
ax.set(title="Monthly net revenue", ylabel="GBP millions"); plt.show()
m.assign(share_of_year=lambda d: d.net_revenue / d.net_revenue.sum()).sort_values("net_revenue", ascending=False).head(5)'''),
    md("""**Business reading.** A gift retailer: revenue climbs from September, peaks in Q4 (Christmas) and falls back in January. The final point is low because **Dec 2011 is incomplete** (data ends 9 Dec) - a classic trap. Plan stock, staff and marketing backwards from the peak."""),
    md("## 3. Where does the money come from? (country revenue)"),
    py('''cr = eda.country_revenue(clean)
top = cr.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.barh(top["Country"], top["net_revenue"] / 1e6, color="#0d9488")
for y, (v, s) in enumerate(zip(top["net_revenue"] / 1e6, top["share"])): ax.text(v, y, f"  {s:.1%}", va="center", fontsize=9)
ax.set(title="Top 10 countries by net revenue", xlabel="GBP millions"); plt.show()'''),
    md("""**Reading.** One market dominates. That is a risk (dependence) and an opportunity (export growth), and it means *Country* will carry little signal in a model - one reason we left it out of the segmentation features."""),
    md("## 4. Best-selling products (top 10 bar chart)"),
    py('''tp = eda.top_products(clean, 10, by="revenue").iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.barh(tp["Description"].str.title().str.slice(0, 38), tp["revenue"] / 1e3, color="#4f46e5")
ax.set(title="Top 10 products by revenue", xlabel="GBP thousands"); plt.show()
eda.top_products(clean, 5, by="units")[["Description", "units", "orders"]]'''),
    md("""Top by **revenue** and top by **units** differ: cheap items sell in bulk, expensive items earn more per unit. Always say which one you mean."""),
    md("## 5. Customers: purchasing frequency and spend (histograms)"),
    py('''orders = eda.customer_order_counts(clean)
spend = sales[sales.HasCustomer].groupby("CustomerID")["Revenue"].sum()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(orders.clip(upper=40), bins=40, color="#2563eb"); ax[0].set(title="Orders per customer (clipped at 40)", xlabel="orders", ylabel="customers")
ax[1].hist(np.log10(spend.clip(lower=1)), bins=40, color="#0d9488"); ax[1].set(title="Total spend per customer (log10 scale)", xlabel="log10(GBP)")
plt.tight_layout(); plt.show()
pd.DataFrame({"orders per customer": eda.distribution_stats(orders), "spend per customer (GBP)": eda.distribution_stats(spend)})'''),
    md("""**Reading the table.** For both variables the **mean is far above the median** and **skew is well above 1**: a long right tail of very loyal, very valuable customers. Consequences we will act on:
* use the **median** (not the mean) to describe a "typical" customer;
* apply a **log transform** before K-Means, otherwise the biggest spenders dominate the distance calculation;
* expect revenue to be **concentrated** in a minority of customers: the next cell shows that the top 20% of customers produce roughly three-quarters of customer revenue."""),
    py('''s = spend.sort_values(ascending=False)
cum = s.cumsum() / s.sum()
for pct in (0.01, 0.05, 0.10, 0.20):
    print(f"top {pct:>4.0%} of customers produce {cum.iloc[int(len(s)*pct)-1]:.1%} of customer revenue")'''),
    md("## 6. Transaction quantity and box plots"),
    py('''fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(np.log10(sales["Quantity"]), bins=40, color="#4f46e5"); ax[0].set(title="Quantity per line (log10 scale)", xlabel="log10(units)")
inv = eda.weekday_hour_pattern(clean)
top5 = inv["revenue"].groupby(clean.drop_duplicates("InvoiceNo").set_index("InvoiceNo")["Country"]).sum().nlargest(5).index
box_df = inv.merge(clean.drop_duplicates("InvoiceNo")[["InvoiceNo", "Country"]], on="InvoiceNo")
box_df = box_df[box_df.Country.isin(top5) & (box_df.revenue > 0)]
sns.boxplot(data=box_df, x="Country", y="revenue", ax=ax[1], color="#93c5fd", fliersize=1.5); ax[1].set_yscale("log")
ax[1].set(title="Invoice value by country (log scale)", ylabel="GBP"); ax[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.show()
print(pd.Series(eda.distribution_stats(sales["Quantity"])).round(2).to_frame("Quantity per line"))'''),
    md("""**Box-plot reading.** The box (middle 50%) is small and sits low; the dots above it are the outliers - bulk wholesale orders. We **flag** very large lines (`IsOutlier`) rather than delete them: they are real customers, and the log transform plus robust models handle them."""),
    md("## 7. Daily purchasing patterns"),
    py('''inv["weekday"] = pd.Categorical(inv["weekday"], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)
heat = inv.pivot_table(index="weekday", columns="hour", values="InvoiceNo" if "InvoiceNo" in inv else "revenue", aggfunc="count", observed=False)
fig, ax = plt.subplots(figsize=(10, 3.6))
sns.heatmap(heat, cmap="Blues", ax=ax, cbar_kws={"label": "orders"}); ax.set(title="When do customers order? (orders by weekday and hour)", xlabel="hour of day", ylabel="")
plt.show()'''),
    md("""**Reading.** About two-thirds of orders are placed between 10:00 and 15:00, and **Saturday is almost empty** (only a few dozen invoices in two years) while Sunday is busy - the shop evidently does not process orders on Saturdays. Together with the huge baskets this suggests a substantial B2B (wholesale) customer base, not just consumers."""),
    md("## 8. Correlation heatmap"),
    py('''corr = eda.correlation_frame(clean)
fig, ax = plt.subplots(figsize=(5.6, 4.4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax); ax.set_title("Spearman correlation (sale lines)")
plt.show()'''),
    md("""**Reading (values from this run).** `Revenue` is driven mainly by `Quantity` (about +0.69) and less by `UnitPrice` (about +0.30). `Quantity` and `UnitPrice` are **negatively** related (about -0.43): expensive items are bought in smaller numbers, cheap ones in bulk. `Hour` has a small negative link with quantity (about -0.18): later-in-the-day orders are slightly smaller. `Month` is near zero - which does **not** mean "no seasonality": correlation only measures *monotonic* relationships, and a Q4 peak is not monotonic in the month number.

## Check yourself
1. Why can the *median* order value be much lower than the *average* order value?
2. A colleague says "revenue crashed in December 2011". What is your reply?
3. Which transformation would you apply to `spend` before clustering, and why?"""),
]

# =====================================================================================================
NB3 = [
    md(briefing("Notebook 3 - Cleaning and customer feature engineering",
        "Show what the cleaning step changed, then turn ~1 million transaction lines into ONE ROW PER CUSTOMER (RFM + behaviour features).",
        "Segmentation and prediction models need a table with one row per customer. Raw data has one row per product line.",
        "'Who are our best customers and who is about to leave?' needs a number for each customer: how recently, how often, how much they buy.",
        "`clean_transactions` (before/after report) then `build_customer_features` aggregates with `groupby` at a **snapshot date**, using only earlier data.",
        "A table of ~5,900 customers x 14 features; a proof that features never look past the snapshot date.",
        "**Data leakage**: accidentally using information from after the prediction moment. It produces beautiful metrics that collapse in production.",
        "They define the observation window and snapshot date explicitly, build features through one function, and unit-test it for leakage (see `tests/test_features.py`).")),
    py(SETUP + '''
from src.features.customer_features import build_customer_features, FEATURE_COLUMNS, validate_features
from src.features.rfm import compute_rfm, add_rfm_scores, default_snapshot
clean = pd.read_parquet(paths.processed / "cleaned_transactions.parquet")
report = json.loads((paths.processed / "cleaning_report.json").read_text())'''),
    md("## Part A - Cleaning: before and after"),
    py('''steps = pd.DataFrame(report["steps"])
steps["rows_removed"] = steps["rows_removed"].map("{:,}".format)
display(steps[["step", "rows_removed", "note"]])
b, a = report["before"], report["after"]
print(f"rows: {report['summary']['rows_before']:,} -> {report['summary']['rows_after']:,}")'''),
    py('''s = report["summary"]
pd.Series({"Gross sales (GBP)": s["gross_sales"], "Returns (GBP)": s["returns_value"], "NET revenue (GBP)": s["net_revenue"],
           "Return rate (% of gross)": s["return_rate_pct"], "Guest rows (%)": s["guest_row_pct"], "Guest revenue (%)": s["guest_revenue_pct"],
           "Flagged outlier lines": s["flagged_outlier_rows"]}).to_frame("value")'''),
    md("""### How returns are treated (the key design decision)
* `RecordType = "sale"` - positive quantity. `RecordType = "return"` - cancellation invoice (`C...`) with negative quantity.
* `Revenue = Quantity x UnitPrice` is therefore **negative for returns**.
* **Net revenue = sum of Revenue.** **Frequency counts only sale invoices** (a return is not a purchase).

Quick proof with one real customer who returned goods:"""),
    py('''has_ret = clean[(clean.RecordType == "return") & clean.HasCustomer]["CustomerID"].value_counts().index[0]
c = clean[clean.CustomerID == has_ret]
print("customer", has_ret)
print(c.groupby("RecordType")["Revenue"].agg(["count", "sum"]))
print("NET spend =", round(c["Revenue"].sum(), 2))'''),
    md("""## Part B - Feature engineering

**Feature engineering** = inventing informative columns from raw data. **Aggregation** with `groupby` collapses many rows into one per group:

```python
sales.groupby("CustomerID")["InvoiceNo"].nunique()   # frequency: distinct orders per customer
```

### RFM - the classic customer triple
| Letter | Meaning | Good is... |
|---|---|---|
| **R**ecency | days since last purchase | small |
| **F**requency | number of distinct orders | large |
| **M**onetary | net spend (sales - returns) | large |

### Observation window and snapshot date (documented)
* **Observation window:** all transactions *before* the snapshot date.
* **Snapshot date:** for the final customer table, the day after the last transaction in the data. For the prediction model, earlier cut-off dates are used (Notebook 5).
* **Leakage guard:** `history_before()` keeps only rows with `InvoiceDate < snapshot_date` (strictly before)."""),
    py('''snapshot = default_snapshot(clean)
print("snapshot date:", snapshot.date())
feats = build_customer_features(clean, snapshot)
assert validate_features(feats) == []
print(f"{len(feats):,} customers x {len(FEATURE_COLUMNS)} features")
feats.head()'''),
    md("""| Feature | Meaning |
|---|---|
| `recency_days`, `frequency`, `monetary` | RFM |
| `avg_order_value` | gross spend per order (GBP) |
| `total_items` | units bought |
| `product_diversity` | distinct products bought |
| `active_months` | distinct calendar months with a purchase |
| `avg_basket_size` | distinct products per order |
| `spend_30d / 60d / 90d` | net spend in the last 30/60/90 days before the snapshot |
| `avg_days_between_purchases` | mean gap between orders (undefined = NaN for one-time buyers) |
| `tenure_days` | days since first purchase |
| `return_rate` | returned value / gross sales |"""),
    md("### Verify by hand (rule: validate every transformation)"),
    py('''cid = int(feats.index[100])
mine = clean[(clean.CustomerID == cid) & (clean.RecordType == "sale") & (clean.InvoiceDate < snapshot)]
print("customer", cid)
print("frequency by hand:", mine.InvoiceNo.nunique(), "| table:", feats.loc[cid, "frequency"])
print("recency   by hand:", (snapshot - mine.InvoiceDate.max()).days, "| table:", feats.loc[cid, "recency_days"])
assert mine.InvoiceNo.nunique() == feats.loc[cid, "frequency"] and (snapshot - mine.InvoiceDate.max()).days == feats.loc[cid, "recency_days"]
saved = pd.read_parquet(paths.processed / "customer_features.parquet")
pd.testing.assert_frame_equal(feats, saved); print("identical to data/processed/customer_features.parquet")'''),
    md("""## Part C - Leakage: see it, then see it prevented
We pretend today is **13 June 2011** and build features. Then we delete every later transaction and rebuild. A leak-free function must return *identical* numbers."""),
    py('''cutoff = pd.Timestamp("2011-06-13")
f_full = build_customer_features(clean, cutoff)
f_past = build_customer_features(clean[clean.InvoiceDate < cutoff], cutoff)
pd.testing.assert_frame_equal(f_full, f_past)
print("OK - identical. Features at", cutoff.date(), "use only the past.")
print(f"customers known on that day: {len(f_full):,}  (vs {len(feats):,} at the end)")
both = f_full[["recency_days", "frequency", "monetary"]].join(feats[["recency_days", "frequency", "monetary"]], rsuffix="_end", how="inner").head(3)
both'''),
    md("""The same customer looks very different on different days - that is exactly what makes a **future-looking label** possible in Notebook 5."""),
    md("## Part D - RFM scores and distributions"),
    py('''scored = add_rfm_scores(compute_rfm(clean, snapshot))
print(scored.head(3))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, col, c in zip(ax, ["recency_days", "frequency", "monetary"], ["#2563eb", "#0d9488", "#4f46e5"]):
    a.hist(feats[col].clip(lower=0), bins=40, color=c); a.set_title(col + " (raw)")
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, col, c in zip(ax, ["recency_days", "frequency", "monetary"], ["#2563eb", "#0d9488", "#4f46e5"]):
    a.hist(np.log1p(feats[col].clip(lower=0)), bins=40, color=c); a.set_title("log1p(" + col + ")")
plt.tight_layout(); plt.show()'''),
    md("""**RFM scores** rank each customer 1-5 on each letter (5 = best). `555` = best customer. They are a simple, explainable baseline; K-Means in Notebook 4 finds groups without hand-written rules.

The **log transform** turns the long right tail into a roughly bell-shaped curve - the shape K-Means handles well."""),
    py('''corr = feats[FEATURE_COLUMNS].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8.5, 6.5)); sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax)
ax.set_title("Feature correlations (Spearman)"); plt.show()'''),
    md("""Many features are strongly correlated (`frequency`, `total_items`, `monetary`). For **K-Means** we therefore use only R, F and M (no double counting). **Tree models** and **XGBoost** cope with correlated features, and the others use regularisation.

## Check yourself
1. Why is `<` (strictly before) used instead of `<=` for the snapshot?
2. Why does `avg_days_between_purchases` have missing values, and who fills them?
3. What would go wrong if `monetary` were computed from *all* data but the label from the next 90 days?"""),
]

# =====================================================================================================
NB4 = [
    md(briefing("Notebook 4 - Customer segmentation with K-Means",
        "Group customers who behave alike - without anyone telling the algorithm what the groups are - then give each group a business name.",
        "Marketing cannot write 5,900 personal campaigns, but it can run 4-6 targeted ones.",
        "Which customers deserve VIP treatment, which are slipping away, which have not yet become loyal?",
        "log1p -> StandardScaler -> K-Means for k=2..10, picks k with silhouette score, names clusters by matching them to business archetypes, projects to 2-D with PCA.",
        "A small number of segments (the data decides how many, up to 6), profile tables and six charts.",
        "Forgetting to scale/log (one feature dominates); trusting the elbow blindly; naming clusters before looking at them; forcing six clusters the data does not contain.",
        "They check cluster stability across random seeds, profile every cluster in original business units, and re-validate segments as new data arrives.")),
    md("""## Concepts

* **Supervised learning** - learn from examples *with* an answer column (Notebook 5: "was the customer high value?"). **Unsupervised learning** - no answer column; find structure (this notebook).
* **K-Means** - pick *k* centre points; assign every customer to the nearest centre; move each centre to the average of its customers; repeat until nothing moves.
* **Centroid** - the centre of a cluster: the "average customer" of that group.
* **Inertia** - total squared distance from every customer to its centroid. Lower = tighter clusters. It *always* falls as k grows, so the **elbow method** looks for the k after which the improvement flattens.
* **Silhouette score** (-1..+1) - for each customer, compares "how close am I to my own cluster" with "how close am I to the next cluster". Near 1 = well separated; near 0 = overlapping; negative = probably in the wrong cluster.
* **PCA (Principal Component Analysis)** - squashes many dimensions into 2 that keep as much spread (variance) as possible, so we can *draw* the clusters. It is for plotting only; clustering happens in the full space."""),
    py(SETUP + '''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from src.models.customer_segmentation import (SEGMENT_FEATURES, prepare_matrix, evaluate_k, choose_k, fit_segmentation, assign_segments, segment_profile, ARCHETYPES, SEGMENT_PLAYBOOK)
from src.utils.config import SEGMENT_COLORS
feats = pd.read_parquet(paths.processed / "customer_features.parquet")
print(f"{len(feats):,} customers; clustering on {SEGMENT_FEATURES}")'''),
    md("## Steps 1-4: choose features, look at distributions, log-transform, scale"),
    py('''x_log = prepare_matrix(feats)
scaler = StandardScaler().fit(x_log); x = scaler.transform(x_log)
fig, ax = plt.subplots(3, 3, figsize=(12, 7))
for j, col in enumerate(SEGMENT_FEATURES):
    ax[0, j].hist(feats[col].clip(lower=0), bins=40, color="#94a3b8"); ax[0, j].set_title(f"{col} - raw")
    ax[1, j].hist(x_log[:, j], bins=40, color="#2563eb"); ax[1, j].set_title("after log1p")
    ax[2, j].hist(x[:, j], bins=40, color="#0d9488"); ax[2, j].set_title("after scaling (mean 0, std 1)")
plt.tight_layout(); plt.show()
print("scaled means:", x.mean(0).round(3), " stds:", x.std(0).round(3))'''),
    md("""Why scale? K-Means measures **distance**. Frequency (1-200) and monetary (GBP 0-500,000) live on different scales; without scaling, money would decide everything. After `log1p` + `StandardScaler`, each feature has mean 0 and standard deviation 1 and counts equally."""),
    md("## Steps 5-7: how many clusters? Elbow + silhouette"),
    py('''metrics = evaluate_k(x)
best_k = choose_k(metrics)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(metrics.k, metrics.inertia, "o-", color="#2563eb"); ax[0].set(title="Elbow curve", xlabel="k (clusters)", ylabel="inertia")
ax[1].plot(metrics.k, metrics.silhouette, "o-", color="#0d9488"); ax[1].axvline(best_k, ls="--", color="#eb6834"); ax[1].set(title="Silhouette score", xlabel="k", ylabel="score")
ax[1].annotate(f"chosen k = {best_k}", (best_k, metrics.set_index("k").silhouette[best_k]), xytext=(8, 8), textcoords="offset points")
plt.tight_layout(); plt.show()
metrics.round(3)'''),
    md("""**How we decide (documented rule in `choose_k`):** among k = 3..6 take the best silhouette; if a larger k is within 0.02 of it, prefer the larger k (a bit more actionable). We stop at 6 because we only have six business labels - and **we never force six**: if the data prefers fewer, we use fewer.

**Be honest about the numbers.** A silhouette around 0.3-0.4 means *moderately* separated groups. Real customers form a continuum (a smear, not islands). Segments are a **useful simplification**, not natural law."""),
    md("## Steps 8-9: train K-Means and read the centroids"),
    py('''bundle = fit_segmentation(feats)
k = bundle["k"]; km = bundle["kmeans"]
print("chosen k:", k, "| silhouette:", round(bundle["silhouette"], 3), "| inertia:", round(km.inertia_, 1))
# centroids live in scaled-log space; undo the steps to read them in business units
centroids = np.expm1(bundle["scaler"].inverse_transform(km.cluster_centers_))
cent = pd.DataFrame(centroids, columns=SEGMENT_FEATURES).rename_axis("cluster")
cent["customers"] = np.bincount(km.labels_)
cent'''),
    md("""Each row is the "average customer" of a cluster (geometric-mean style, because we averaged in log space). Compare `recency_days`: a small number means the customers bought recently."""),
    md("## Steps 10-11: profiles and business labels"),
    py('''seg = assign_segments(bundle, feats)
profile = segment_profile(feats, seg["segment"])
print("cluster -> label:", bundle["cluster_to_name"])
profile[["customers", "customer_share", "recency_median", "frequency_median", "monetary_median", "avg_order_value", "tenure_median", "revenue_share"]]'''),
    md("""**How labels are assigned (no hand-picking):** each business label has an *archetype* - target percentile scores for recency, frequency, monetary and tenure (e.g. Champions = very recent, very frequent, very high spend). We compare each cluster's median profile with every archetype and use the **Hungarian algorithm** to give each cluster the best *distinct* label. With 4 clusters, only the 4 best-fitting labels are used.

| Label | Archetype |
|---|---|
| Champions | recent, frequent, high spend |
| Loyal Customers | frequent, solid spend, fairly recent |
| Potential Loyalists | recent, only a few orders so far |
| New Customers | very recent *first* purchase, short tenure |
| At-Risk Customers | used to spend a lot, not seen for a long time |
| Hibernating Customers | low spend, long gone |"""),
    py('''fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
order = profile.sort_values("revenue_share", ascending=False).index
colors = [SEGMENT_COLORS.get(s, "#64748b") for s in order]
ax[0].bar(order, profile.loc[order, "customer_share"] * 100, color=colors); ax[0].set(title="Share of customers (%)"); ax[0].tick_params(axis="x", rotation=25)
ax[1].bar(order, profile.loc[order, "revenue_share"] * 100, color=colors); ax[1].set(title="Share of net revenue (%)"); ax[1].tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.show()'''),
    md("""**Business reading.** Compare the two bars: the segment that is a small share of customers but a *large* share of revenue is where retention effort pays most; the segment that is a big share of customers but a tiny share of revenue needs cheap, automated treatment."""),
    py('''# Segment characteristics heatmap: each cell = percentile rank (0-1) of the segment's median vs ALL customers
cols = ["recency_days", "frequency", "monetary", "avg_order_value", "product_diversity", "tenure_days"]
med = feats.assign(segment=seg["segment"]).groupby("segment")[cols].median()
pct = pd.DataFrame({c: [(feats[c] <= v).mean() for v in med[c]] for c in cols}, index=med.index)
pct["recency_days"] = 1 - pct["recency_days"]                # flip: 1 = bought very recently
pct = pct.rename(columns={"recency_days": "recency (1 = recent)"}).loc[order]
fig, ax = plt.subplots(figsize=(8.5, 3.4)); sns.heatmap(pct, annot=True, fmt=".2f", cmap="GnBu", vmin=0, vmax=1, ax=ax)
ax.set(title="Segment characteristics (percentile of the whole customer base)", ylabel=""); plt.show()'''),
    md("## PCA: draw the clusters in 2-D"),
    py('''pca = bundle["pca"]
print("variance explained by the 2 components:", pca.explained_variance_ratio_.round(3), "-> total", pca.explained_variance_ratio_.sum().round(3))
fig, ax = plt.subplots(figsize=(8.5, 5.4))
for s in order:
    m = seg["segment"] == s
    ax.scatter(seg.loc[m, "pc1"], seg.loc[m, "pc2"], s=9, alpha=.55, color=SEGMENT_COLORS.get(s, "#64748b"), label=f"{s} ({m.sum():,})")
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", title="Customers in PCA space")
ax.legend(markerscale=2.5, frameon=False); plt.show()
pd.DataFrame(pca.components_, columns=SEGMENT_FEATURES, index=["PC1", "PC2"]).round(2)'''),
    md("""The loadings table says what each axis *means* (e.g. a PC that weights frequency and monetary positively and recency negatively is a "how valuable" axis). Because the groups touch, you see the continuum mentioned earlier.

## Silhouette analysis (per customer)"""),
    py('''rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(x), min(3000, len(x)), replace=False)
sil = silhouette_samples(x[idx], km.labels_[idx])
fig, ax = plt.subplots(figsize=(8, 4.2)); y0 = 0
for c in range(k):
    v = np.sort(sil[km.labels_[idx] == c]); name = bundle["cluster_to_name"][c]
    ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=SEGMENT_COLORS.get(name, "#64748b"), alpha=.85); ax.text(-0.18, y0 + len(v) / 2, name, va="center", fontsize=8); y0 += len(v) + 40
ax.axvline(sil.mean(), color="#eb6834", ls="--"); ax.set(title=f"Silhouette per customer (mean {sil.mean():.2f})", xlabel="silhouette", yticks=[]); plt.show()
print("customers with a NEGATIVE silhouette (probably borderline):", f"{(sil < 0).mean():.1%}")'''),
    md("""## Is the result stable? (what professionals check)
K-Means starts from random centres. If a different seed gives a very different answer, the segments are not trustworthy. **Adjusted Rand Index (ARI)** = 1 means identical groupings."""),
    py('''ari = [adjusted_rand_score(km.labels_, KMeans(n_clusters=k, n_init=10, random_state=s).fit(x).labels_) for s in range(1, 6)]
print("ARI vs the final model across 5 other seeds:", np.round(ari, 3))'''),
    py('''import joblib
saved = joblib.load(paths.models / "kmeans_customer_segments.joblib")
assert saved["k"] == k and (assign_segments(saved, feats)["cluster"] == seg["cluster"]).all()
print("Saved model models/kmeans_customer_segments.joblib reproduces these segments exactly.")
for s in order: print(f"\\n{s}: {SEGMENT_PLAYBOOK[s]['who']}\\n   -> {SEGMENT_PLAYBOOK[s]['action']}")'''),
    md("""## Check yourself
1. Why would K-Means without scaling produce a "money only" clustering?
2. The elbow is not obvious here. Which second tool do we use, and what does a score of 0.35 mean?
3. If the data supports only 4 clusters, why is that *better* than forcing six?"""),
]

# =====================================================================================================
NB5 = [
    md(briefing("Notebook 5 - Predicting customer value",
        "Train five classifiers to predict, today, which customers will be in the top 20% by spend over the **next 90 days**.",
        "Segmentation describes the past; prediction lets marketing act *before* value is realised.",
        "Where should the retention budget go? Contact the customers most likely to spend a lot next quarter.",
        "Builds three time-separated datasets (train/validation/test), tunes five models on validation data, picks one by a documented business rule, then scores the untouched test period once.",
        "A comparison table, ROC/PR curves, a confusion matrix, feature importance. Expect good-but-not-perfect numbers (ROC-AUC mid-0.8s).",
        "Leakage; random (non-temporal) splits; tuning on the test set; reporting accuracy on imbalanced data; trusting a single number.",
        "They split by time, tune only on validation, choose a threshold from business costs, report several metrics and monitor drift after launch.")),
    md("""## The design in one picture

```
 train cutoff T1 --90 days--> validation cutoff T2 --90 days--> test cutoff T3 --90 days--> end of data
 features: past < T1           features: past < T2               features: past < T3
 label:    spend in T1..T2     label:    spend in T2..T3         label:    spend in T3..end
```
* **Features** come only from *before* each cut-off. **Labels** come only from the 90 days *after* it.
* The model learns on T1, is **tuned and selected** on T2, and is judged **once** on T3.
* **Label rule (documented):** a customer is *high value* if their **net revenue in the next 90 days is in the top 20%** of customers known on the cut-off date (a per-window quantile keeps the share comparable between a quiet summer and a Christmas peak). Customers who do not buy again have revenue 0 and are labelled 0.
* The label is **not** one of the input features, so the model cannot cheat.

## Concepts
* **Classification** - predict a category (here: high value yes/no) as a probability.
* **Train / validation / test** - train = learn; validation = choose settings and the winning model; test = one final honest exam.
* **Overfitting** - memorising training noise: great on train, worse on new data. **Underfitting** - too simple to see the pattern: poor on both.
* **Cross-validation** - repeat train/validate on several slices to see how stable a score is. We report 5-fold CV on the *training* period as a sanity check; because time matters, selection uses the later validation window.
* **Precision** - of customers we flag, how many are truly high value? **Recall** - of all truly high-value customers, how many did we flag? **F1** - their harmonic mean.
* **ROC-AUC** - probability that a random high-value customer is scored above a random other one (0.5 = coin flip, 1 = perfect). **PR-AUC** is more informative when positives are rare.
* **Accuracy** can mislead: with 20% positives, saying "nobody is high value" is already 80% accurate."""),
    py(SETUP + '''
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from src.features.customer_features import FEATURE_COLUMNS
from src.models.value_prediction import make_windows, build_split, make_preprocessor, train_and_select
from src.models.rbf_network import RBFNetworkClassifier
clean = pd.read_parquet(paths.processed / "cleaned_transactions.parquet")'''),
    md("## 1. Build the three time windows"),
    py('''w = make_windows(clean)
print(pd.Series(w.as_dict()).to_frame("value").T.to_string())
train, val, test = (build_split(clean, c, w.horizon) for c in (w.train, w.val, w.test))
summary = pd.DataFrame({"cutoff": [w.train, w.val, w.test], "customers": [len(d) for d in (train, val, test)],
                        "positive rate": [d.high_value.mean() for d in (train, val, test)],
                        "GBP needed to be 'high value'": [d.attrs["threshold_gbp"] for d in (train, val, test)]}, index=["train", "validation", "test"])
summary'''),
    md("""The GBP threshold is higher in the test window - it is the Christmas quarter, so everybody spends more. That is exactly why the label is a *quantile* (top 20%) and not a fixed amount. Note this is a **distribution shift**: models trained on a quiet quarter must still work in the peak."""),
    py('''# Leakage audit: nothing derived from the future may be a feature
assert "future_revenue" not in FEATURE_COLUMNS and "high_value" not in FEATURE_COLUMNS
print("features given to the model:", FEATURE_COLUMNS)
print("correlation of each feature with the label (train):")
train[FEATURE_COLUMNS + ["high_value"]].corr(method="spearman")["high_value"].drop("high_value").sort_values(ascending=False).round(2).to_frame("spearman")'''),
    md("""No feature is *suspiciously* perfect (a correlation near 1 would be a leakage alarm). Recent spend and monetary value predict the future moderately - as common sense says."""),
    md("""## 2. Underfitting vs overfitting - see it with your own eyes
A random forest with `max_depth` from 1 (too simple) to 25 (very flexible). Watch train vs validation ROC-AUC."""),
    py('''prep = make_preprocessor().fit(train[FEATURE_COLUMNS])
Xtr, Xva = prep.transform(train[FEATURE_COLUMNS]), prep.transform(val[FEATURE_COLUMNS])
rows = []
for d in [1, 2, 3, 4, 6, 8, 12, 18, 25]:
    rf = RandomForestClassifier(n_estimators=150, max_depth=d, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, train.high_value)
    rows.append({"max_depth": d, "train AUC": roc_auc_score(train.high_value, rf.predict_proba(Xtr)[:, 1]), "validation AUC": roc_auc_score(val.high_value, rf.predict_proba(Xva)[:, 1])})
gap = pd.DataFrame(rows)
fig, ax = plt.subplots(); ax.plot(gap.max_depth, gap["train AUC"], "o-", label="train", color="#2563eb"); ax.plot(gap.max_depth, gap["validation AUC"], "o-", label="validation", color="#eb6834")
ax.set(xlabel="model complexity (max_depth)", ylabel="ROC-AUC", title="The overfitting gap"); ax.legend(frameon=False); plt.show()
gap.round(3)'''),
    md("""Left: both curves low = **underfitting**. Right: train near 1.0 while validation stalls or drops = **overfitting** (the gap). We want the validation curve's peak, which is what tuning on validation data finds."""),
    md("""## 3. Train, tune and compare five models
`train_and_select` does everything in one reproducible call (this is the same code the pipeline uses):

| Model | Idea |
|---|---|
| Logistic Regression | a weighted sum of features turned into a probability; fast, explainable baseline |
| Random Forest | hundreds of decision trees voting |
| XGBoost | trees built one after another, each fixing the last one's errors |
| **RBF Network** | hidden neurons = prototype customers; each measures closeness with a Gaussian; a logistic output layer combines them |
| MLP Classifier | a standard feed-forward neural network |

Hyper-parameters (tree depth, `gamma`, `C` ...) are searched on the **validation** set; the decision threshold is the one that maximises F1 on validation; the **test** period is only scored afterwards."""),
    py('''res = train_and_select(clean, fast=False)
metrics = res["metrics"]
print("SELECTED:", metrics["selected_model"]); print("RULE    :", metrics["selection_rule"])
cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
table = lambda split: pd.DataFrame({n: {c: r[split][c] for c in cols} for n, r in metrics["models"].items()}).T
print("\\nVALIDATION (used to choose)"); display(table("validation"))
print("TEST (touched once)");       display(table("test"))'''),
    py('''best_params = pd.DataFrame({n: r["best_params"] for n, r in metrics["models"].items()}).T
cv = pd.DataFrame({n: {"5-fold CV ROC-AUC (train period)": r["cv_roc_auc_train_mean"], "std": r["cv_roc_auc_train_std"]} for n, r in metrics["models"].items()}).T
display(best_params); display(cv)
meta = json.loads((paths.models / "run_metadata.json").read_text())
assert meta["selected_model"] == metrics["selected_model"]; print("Same model as the one saved by the pipeline - results are reproducible.")'''),
    md("""**How to read it.** The models are close: for *tabular* data with few features, simple models are often as good as complex ones. Differences of 0.01 in AUC are within noise. We therefore pick with a **business rule**, not by a tiny edge: highest validation PR-AUC among models that reach a minimum precision (we do not want to spam non-valuable customers) and recall (we do not want to miss most valuable ones)."""),
    md("## 4. ROC curves, precision-recall curves, confusion matrix (test period)"),
    py('''fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
pal = dict(zip(metrics["models"], ["#64748b", "#0d9488", "#2563eb", "#4f46e5", "#eb6834"]))
for n, r in metrics["models"].items():
    lw = 3 if n == metrics["selected_model"] else 1.4
    ax[0].plot(r["test"]["roc_curve"]["fpr"], r["test"]["roc_curve"]["tpr"], lw=lw, color=pal[n], label=f"{n} ({r['test']['roc_auc']:.3f})")
    ax[1].plot(r["test"]["pr_curve"]["recall"], r["test"]["pr_curve"]["precision"], lw=lw, color=pal[n], label=f"{n} ({r['test']['pr_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set(title="ROC curve", xlabel="false positive rate", ylabel="true positive rate")
ax[1].axhline(res["data"]["test"].high_value.mean(), color="k", ls=":", lw=1); ax[1].set(title="Precision-recall curve (dotted = base rate)", xlabel="recall", ylabel="precision")
for a in ax: a.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()'''),
    py('''sel = metrics["selected_model"]; cm = np.array(metrics["models"][sel]["test"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.8, 4)); sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, xticklabels=["not high", "high"], yticklabels=["not high", "high"], ax=ax)
ax.set(title=f"{sel}: confusion matrix (test)", xlabel="predicted", ylabel="actual"); plt.show()
tn, fp, fn, tp = cm.ravel()
print(f"flagged {tp+fp:,} customers: {tp:,} truly high value (precision {tp/(tp+fp):.0%}); missed {fn:,} (recall {tp/(tp+fn):.0%}).")'''),
    md("""**Business translation.** A confusion matrix has four cells: true positives (right flags), false positives (wasted outreach), false negatives (missed opportunities), true negatives. The decision threshold moves the balance between wasted outreach and missed opportunities - choose it from the *cost* of each mistake, not from a default of 0.5."""),
    md("""## 5. The RBF network under the microscope
The RBF network is built from scratch in `src/models/rbf_network.py`. The key hyper-parameter `gamma` is the **width of each Gaussian**: large gamma = each prototype only "sees" very close customers (risk of overfitting); small gamma = wide, smooth, blurry (risk of underfitting)."""),
    py('''ytr, yva = train.high_value.to_numpy(), val.high_value.to_numpy()
gammas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
out = []
for g in gammas:
    m = RBFNetworkClassifier(n_centers=25, gamma=g, C=1.0, random_state=RANDOM_STATE).fit(Xtr, ytr)
    out.append({"gamma": g, "train PR-AUC": average_precision_score(ytr, m.predict_proba(Xtr)[:, 1]), "validation PR-AUC": average_precision_score(yva, m.predict_proba(Xva)[:, 1])})
gam = pd.DataFrame(out)
fig, ax = plt.subplots(); ax.semilogx(gam.gamma, gam["train PR-AUC"], "o-", label="train", color="#2563eb"); ax.semilogx(gam.gamma, gam["validation PR-AUC"], "o-", label="validation", color="#eb6834")
ax.set(xlabel="gamma (log scale)", ylabel="PR-AUC", title="RBF width: tuning gamma on validation data"); ax.legend(frameon=False); plt.show()
gam.round(3)'''),
    md("""**What the curve actually shows (read it, do not assume it).**
* Performance peaks at a moderate `gamma` (a few hundredths) and **collapses for large gamma on both train and validation**. With `gamma` too large each Gaussian is so narrow that almost every customer is "far" from every prototype, all activations are close to 0 and the output layer has nothing to work with - this is **underfitting caused by vanishing activations**, not the classic memorisation overfit. (Overfitting would show train rising while validation falls; with only 25 centres and a regularised output layer the RBF network cannot memorise much.)
* **Validation scores sit slightly above train scores.** That is not a bug: the two windows are different periods with different customers, and the validation period happens to be a little easier to predict. A train-vs-validation gap with the *wrong* sign is a reminder that a score is only meaningful relative to the period it was measured on.
* Tuning `gamma` (and `C`) on validation data, as the pipeline does, picks the plateau at the top."""),
    py('''m = RBFNetworkClassifier(n_centers=8, gamma=0.05, C=1.0, random_state=RANDOM_STATE).fit(Xtr, ytr)
one = Xva[:1]; act = m._activations(one)[0]
print("One customer's 8 hidden-neuron activations (1 = identical to that prototype, ~0 = far away):")
print(np.round(act, 3))
print("output layer turns these into P(high value) =", round(float(m.predict_proba(one)[0, 1]), 3))'''),
    md("## 6. Explainability: which features drive the prediction?"),
    py('''from src.analytics.explain import permutation_table, shap_table
perm = permutation_table(res["bundle"]["pipeline"], res["data"]["val"][FEATURE_COLUMNS], res["data"]["val"].high_value.to_numpy(), n_repeats=8)
shap_df = shap_table(res["fitted"]["XGBoost"], res["prep"], res["data"]["test"][FEATURE_COLUMNS])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
p = perm.head(10).iloc[::-1]; ax[0].barh(p.feature, p.importance, xerr=p["std"], color="#2563eb"); ax[0].set(title="Permutation importance (selected model, validation)", xlabel="drop in PR-AUC when shuffled")
if shap_df is not None:
    s = shap_df.head(10).iloc[::-1]; ax[1].barh(s.feature, s.mean_abs_shap, color="#0d9488"); ax[1].set(title="Mean |SHAP| (XGBoost, test sample)", xlabel="average impact on prediction")
plt.tight_layout(); plt.show()
if shap_df is not None: display(shap_df.head(8).round(3))'''),
    md("""* **Permutation importance** shuffles one feature and measures how much the score falls: a big drop = the model relied on it.
* **SHAP** splits each single prediction into per-feature contributions. `direction_corr` > 0 means "higher value of this feature pushes the prediction *up*"; for `recency_days` a *negative* sign is expected (longer since last order -> lower chance).
* Both are about *what the model uses*, not about *what causes* value.

## Check yourself
1. Why would a random 80/20 split of customers overstate performance here?
2. The test set has a higher GBP threshold than training. Is the model broken?
3. You can only contact 500 customers. How would you choose the threshold?"""),
]

# =====================================================================================================
NB6 = [
    md(briefing("Notebook 6 - Product recommendation engine",
        "Build and compare four recommenders: popularity, collaborative filtering, content-based and a hybrid, and evaluate them without peeking at the future.",
        "Recommendations raise basket size and bring customers back; a poor one just shows best-sellers to everyone.",
        "'Customers who bought this also bought ...' and 'more like this' on the product page, in e-mails and in the basket.",
        "Builds a customer x product matrix, item-item cosine similarity, TF-IDF text vectors, blends them with weights 0.6/0.3/0.1 after min-max normalisation, evaluates on a later time period.",
        "A top-5 list with reasons, and a metrics table. Expect LOW absolute numbers - predicting *new* products is hard - and a hybrid that beats the baselines.",
        "Recommending what the customer already owns; evaluating on data from before the training cut-off; one popular product dominating; cold start; best-seller bias.",
        "They evaluate offline with time splits, then confirm with an online A/B test, and watch coverage and popularity bias - not just precision.")),
    md("""## The four recommenders

| Recommender | How it decides | Strength | Weakness |
|---|---|---|---|
| **Popularity** | most orders overall | works for everyone, strong baseline | not personal; recommends the same items to all |
| **Collaborative filtering (item-based)** | products bought by *the same customers* are similar (cosine similarity of customer columns) | finds non-obvious links; no text needed | needs history; favours popular items; cold start |
| **Content-based** | products with *similar descriptions* (TF-IDF + cosine) | works for rarely-sold products | only "more of the same" |
| **Hybrid** | `0.60 x collaborative + 0.30 x content + 0.10 x popularity` | balances the three | weights need tuning on data |

* **Cosine similarity** = the angle between two vectors (1 = same direction, 0 = unrelated). It ignores how *long* the vectors are, so a popular product is not automatically "more similar".
* **TF-IDF** weights a word by how frequent it is in this description (TF) and how *rare* across all descriptions (IDF). "Heart" is informative; "of" is not.
* **Normalise first:** the three scores live on different scales. Each is min-max scaled to 0-1 over the candidate products *before* blending; otherwise the biggest raw numbers silently win.
* **Cold start:** a customer with no history gets the popularity list (flagged `fallback`)."""),
    py(SETUP + '''
from src.recommendations.common import build_reco_data
from src.recommendations.hybrid import HybridRecommender, WEIGHTS
from src.recommendations.evaluation import evaluate_recommenders
clean = pd.read_parquet(paths.processed / "cleaned_transactions.parquet")
data = build_reco_data(clean)
print(f"{len(data.customer_ids):,} customers x {data.n_items:,} products")'''),
    md("## 1. The customer-product interaction matrix"),
    py('''M = data.matrix
density = M.nnz / (M.shape[0] * M.shape[1])
print(f"non-zero cells: {M.nnz:,}  |  density: {density:.2%}  (i.e. {1-density:.2%} of cells are empty - this is called SPARSITY)")
per_cust = np.asarray(M.sum(1)).ravel(); per_item = np.asarray(M.sum(0)).ravel()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(per_cust, bins=50, color="#2563eb"); ax[0].set(title="Distinct products per customer", xlabel="products")
ax[1].hist(per_item, bins=50, color="#0d9488"); ax[1].set(title="Distinct customers per product", xlabel="customers (long tail = a few blockbusters)")
plt.tight_layout(); plt.show()'''),
    md("## 2. Collaborative filtering - 'customers who bought A also bought B'"),
    py('''rec = HybridRecommender.from_transactions(clean)
pop_order = np.argsort(-data.popularity)
anchor = int(pop_order[10])
sim = rec.cf.sim[anchor]
nearest = np.argsort(-sim)[:5]
print("Anchor product :", data.items.at[anchor, "Description"].title())
print("Most similar by CUSTOMER BEHAVIOUR:")
for j in nearest: print(f"   {sim[j]:.2f}  {data.items.at[j, 'Description'].title()}")'''),
    md("""Expect close relatives (same family or colour variants). Collaborative filtering never read the words - it discovered the family purely from who bought what."""),
    md("## 3. Content-based filtering - 'more like this'"),
    py('''cb = rec.cb
print("type of the TF-IDF matrix:", type(cb.tfidf).__name__, "shape", cb.tfidf.shape)
from sklearn.metrics.pairwise import cosine_similarity
cs = cosine_similarity(cb.tfidf[anchor], cb.tfidf).ravel(); cs[anchor] = 0
print("\\nAnchor product :", data.items.at[anchor, "Description"].title())
print("Most similar by DESCRIPTION TEXT:")
for j in np.argsort(-cs)[:5]: print(f"   {cs[j]:.2f}  {data.items.at[j, 'Description'].title()}")'''),
    md("## 4. Hybrid recommendations for one customer, with explanations"),
    py('''cid = int(data.customer_ids[np.argmax(per_cust)])      # a heavy buyer
print("customer", cid, "has bought", len(data.history[cid]), "different products")
out = rec.recommend(cid, n=5)
display(out[["description", "score", "collaborative", "content", "popularity", "reason"]].round(3))
bought = {data.items.at[i, "StockCode"] for i in data.history[cid]}
assert not set(out.stock_code) & bought; print("none of the recommended products was bought before - OK")'''),
    py('''print("A brand-new visitor (no history) gets the fallback:")
rec.recommend(customer_id=None, n=3)[["description", "reason", "fallback"]]'''),
    md("""## 5. Time-aware evaluation (no peeking)

1. Pick a **split date** (the last 60 days of data).
2. Fit every recommender **only on transactions before it**.
3. The ground truth for a customer is the set of **new** products they bought *after* the split - products they never bought before. (Re-buying the same item would be too easy and not a "recommendation".)
4. Recommend 5 products and compare.

| Metric | Question it answers |
|---|---|
| **Precision@5** | of the 5 shown, how many were bought? |
| **Recall@5** | of everything they went on to buy, how much did we show? |
| **NDCG@5** | did we put the hits at the *top* of the list? |
| **Coverage** | what share of the catalogue ever gets recommended? |
| **Popularity bias** | do we only recommend best-sellers? (share of recommendations from the top-10% most popular products) |"""),
    py('''split = clean["InvoiceDate"].max().normalize() - pd.Timedelta(days=60)
ev = evaluate_recommenders(clean, split)
print("split date:", split.date(), "| evaluation customers:", ev["n_eval_customers"], "| catalogue size at split:", ev["n_train_items"])
tbl = pd.DataFrame(ev["models"]).T
display(tbl.round(4))
saved = json.loads((paths.models / "recommender_eval.json").read_text())
assert saved["models"]["hybrid"] == ev["models"]["hybrid"]; print("matches models/recommender_eval.json")'''),
    py('''fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, col, t in zip(ax, ["precision@5", "ndcg@5", "coverage"], ["Precision@5", "NDCG@5", "Catalogue coverage"]):
    a.bar(tbl.index, tbl[col], color=["#94a3b8", "#2563eb", "#0d9488", "#4f46e5"]); a.set_title(t); a.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.show()
print(f"customers' actual new purchases sit at popularity percentile {ev['actual_purchase_popularity_percentile']:.2f};"
      f" recommendations sit at: " + ", ".join(f"{n} {r['avg_popularity_percentile']:.2f}" for n, r in ev['models'].items()))'''),
    md("""### How to read these honestly
* Absolute precision is **low** (a few percent). That is normal: with thousands of products and a *new-item* target, even a perfect-taste system rarely hits. What matters is the **ranking**: hybrid and the personalised models beat plain popularity.
* **Popularity bias:** compare the "popularity percentile" of recommendations with that of what customers *actually* bought. If recommendations sit much higher, the system over-promotes best-sellers - good for accuracy, bad for discovery. **Coverage** shows the same story from the catalogue side: popularity recommends a sliver of products; content-based reaches far more.
* The hybrid is a **trade-off** (0.6/0.3/0.1 is given by the project brief, not tuned on this test period - tuning it on the same data we report would be cheating). A production team would tune the weights on a *validation* period and confirm with an A/B test.

## Check yourself
1. Why must the recommender be fitted only on data *before* the split date?
2. Why do we normalise each component before blending?
3. A product has been in the catalogue for one week. Which of the three signals can still recommend it?"""),
]


def main(execute: bool) -> None:
    NB_DIR.mkdir(exist_ok=True)
    books = {"01_data_understanding.ipynb": NB1, "02_exploratory_data_analysis.ipynb": NB2, "03_customer_feature_engineering.ipynb": NB3,
             "04_customer_segmentation.ipynb": NB4, "05_customer_value_prediction.ipynb": NB5, "06_recommendation_engine.ipynb": NB6}
    for name, cells in books.items():
        nb = build(name, cells)
        if execute:
            from nbconvert.preprocessors import ExecutePreprocessor
            print("executing", name, flush=True)
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
            nbf.write(nb, NB_DIR / name)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    main(not ap.parse_args().no_exec)
