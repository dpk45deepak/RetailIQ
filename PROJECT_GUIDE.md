# RetailIQ - project guide (read this to own the project)

Written in the first person so you can read it, absorb it and explain the project in your own words. Everything here matches the code and the saved results in this repo; where a number appears, it comes from `models/*.json` or `data/processed/*`.

> **A note on honesty.** This project was built with an AI assistant. In an interview or portfolio, say so, then show that you understand it. This guide exists so you *do*: if you can explain section 5 (decisions) and section 9 (questions) without looking, you own it.

---

## 1. The 60-second pitch

> "RetailIQ analyses about a million transactions from a UK online gift retailer. I cleaned the data - keeping returns and netting them against sales - turned it into one row per customer, and then did three things: **segmented** customers with K-Means (the data supported 4 groups, not the 6 I started with), **predicted** which customers will be in the top 20% by spend over the next 90 days (five models, including an RBF neural network I wrote from scratch, evaluated on a later time period so nothing leaks), and built a **hybrid recommender** (collaborative + content + popularity). It's served through a six-page Streamlit dashboard and a FastAPI service, with 52 tests. The headline finding: 20% of customers - the Champions - produce 73% of revenue, and 1,482 At-Risk customers who spent GBP 2.77M have gone quiet."

---

## 2. The problem and the data

**Business problem.** A retailer has 5,000+ customers and cannot treat them all the same. Marketing wants to know: *who are my best customers, who is leaving, who will be valuable next quarter, and what should I show each person?*

**Data.** UCI *Online Retail II*: 1,067,371 rows, one row per product on an invoice, 1 Dec 2009 - 9 Dec 2011.
Columns: `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`.

**What is wrong with it** (this is the story of Phase 1 - know it by heart):

| Problem | Size | What I did |
|---|---|---|
| Exact duplicate rows | 34,335 | removed |
| Invoices starting with `C` = returns/cancellations | 19,494 rows | **kept as returns**, netted against sales |
| Price <= 0 (free items, adjustments) | ~6,000 | removed |
| Non-product codes (`POST`, `D`, `M`, `BANK CHARGES` ...) | ~5,800 | removed from sales analysis |
| Missing CustomerID (guest checkouts) | 22.8% of rows, ~13.6% of revenue | kept for company totals, excluded from customer models |
| Absurd quantities (e.g. 80,995 units) | 13 lines | removed; 1,002 big-but-plausible lines only *flagged* |

After cleaning: 1,021,241 rows, net revenue **GBP 18.91M**, returns 2.43% of gross sales, 5,852 customers with purchases.

---

## 3. The big picture

```
raw CSV/XLSX
   |  load_data -> validate_data -> cleaning            (Phase 1-3)
   v
cleaned_transactions.parquet   (sale lines + return lines, Revenue = qty x price, negative for returns)
   |  rfm.py + customer_features.py                     (Phase 4)
   v
customer_features.parquet      (1 row per customer, 14 features, snapshot date 2011-12-10)
   |
   +--> customer_segmentation.py  : log -> scale -> K-Means -> PCA -> business labels   (Phase 5)
   +--> value_prediction.py       : 3 time windows, 5 models, pick on validation         (Phase 6)
   +--> recommendations/*         : popularity, item-CF, TF-IDF, hybrid, evaluation      (Phase 7)
   +--> analytics/*               : EDA, SHAP/permutation importance, insights           (Phase 8)
   v
models/*.joblib + *.json, data/processed/*.parquet
   |
   +--> Streamlit dashboard (app.py + pages/)           (Phase 9)
   +--> FastAPI (api/)                                  (Phase 10)
   +--> pytest (52 tests)                               (Phase 11)
```

One command rebuilds everything: `python scripts/run_pipeline.py` (about a minute).
Golden rule I followed: **notebooks explain, `src/` is the real code** - notebooks import the same functions the dashboard and API use.

---

## 4. Phase by phase - what, why, and what to say

### Phase 1 - Understand the data
*What:* loaded the file, checked types, missing values and quality problems (notebook 01).
*Why:* a model learns whatever the data says, including its mistakes.
*Say:* "I profiled before touching anything and wrote down a decision for each problem."

### Phase 2 - EDA
*What:* KPIs, monthly revenue, country revenue, top products, order/spend distributions, weekday-hour heatmap, correlations (notebook 02).
*Findings you can quote:*
* Revenue peaks in Q4 (November is about 1.8x a typical month; Oct-Dec is 35% of annual revenue).
* The UK is 85% of net revenue.
* Spend per customer is extremely right-skewed: the top 20% of customers produce roughly three-quarters of customer revenue. So I use **medians** and **log transforms**.
* Saturday is almost empty and baskets are huge: a lot of B2B/wholesale buyers.
* Quantity and price are negatively correlated (cheap items are bought in bulk).

### Phase 3 - Cleaning (`src/data/cleaning.py`)
The key idea: **do not blindly delete negative quantities.**
* `RecordType = sale` or `return`. `Revenue = Quantity x UnitPrice` (negative for returns).
* **Net revenue** = sum of Revenue. **Frequency** counts only *sale* invoices (a return is not a purchase).
* Every step is logged with row counts (`data/processed/cleaning_report.json`, `docs/data_quality_report.md`).

### Phase 4 - Features (`src/features/`)
One row per customer, built *at a snapshot date* using only rows **strictly before** it.
* RFM: **R**ecency (days since last order), **F**requency (distinct sale invoices), **M**onetary (net spend).
* Extras: average order value, total items, product diversity, active months, basket size, spend in last 30/60/90 days, average days between orders, tenure, return rate.
* **Leakage guard:** a test proves features at a cutoff are identical whether or not later data exists.

### Phase 5 - Segmentation (`src/models/customer_segmentation.py`)
Pipeline: `log1p` -> `StandardScaler` -> K-Means on recency, frequency, monetary, tenure.
* **Choosing k:** evaluated k = 2..10 with elbow and silhouette; rule = best silhouette among k = 3..6, prefer a larger k if within 0.02. Result: **k = 4**, silhouette 0.359 (moderate separation - customers form a continuum).
* **Naming clusters without hand-picking:** each business label (Champions, Loyal, Potential Loyalists, New, At-Risk, Hibernating) has an "archetype" profile of R/F/M/tenure percentiles. I compare each cluster's profile to every archetype and use the **Hungarian algorithm** to assign unique best-fit labels.
* **Stability:** re-ran with 5 other seeds; adjusted Rand index 0.97-1.00.
* **Result:**

| Segment | Customers | Share of revenue | Median recency | Median orders |
|---|---|---|---|---|
| Champions | 1,156 (20%) | **73%** | 16 days | 13 |
| At-Risk | 1,482 (25%) | 17% | 176 days | 5 |
| Potential Loyalists | 1,234 (21%) | 6% | 23 days | 3 |
| Hibernating | 1,980 (34%) | 4% | 400 days | 1 |

### Phase 6 - Predicting customer value (`src/models/value_prediction.py`)
**Question:** *as of a cutoff date, which customers will be in the top 20% by net spend over the next 90 days?*

```
cutoff 2011-03-15 (train) -> 2011-06-13 (validation) -> 2011-09-11 (test) -> data ends 2011-12-10
features: only data BEFORE the cutoff      label: spend in the 90 days AFTER the cutoff
```
* **Label** = top 20% of net spend in the next 90 days *within that window* (a quantile, so quiet summer and Christmas are comparable). Not a feature - so no cheating.
* **Preprocessing** fitted on train only: median imputation (+ missing flags), signed log, scaling.
* **Five models:** logistic regression, random forest, XGBoost, **RBF network (mine)**, MLP. Hyper-parameters and the decision threshold (max F1) are tuned on **validation**. Test is scored once.
* **Selection rule:** highest validation PR-AUC among models with validation precision >= 0.40 and recall >= 0.50 -> **XGBoost**.
* **Test results (Christmas quarter):** XGBoost ROC-AUC 0.868, PR-AUC 0.683, precision 0.55, recall 0.70. The five models are within about 0.01 AUC of each other - so honestly, model choice matters little here; the time-split protocol and features matter more.
* **Explainability:** permutation importance and SHAP; top drivers are monetary value, total items and average order value.

**The RBF network in plain words** (`src/models/rbf_network.py`, about 50 lines):
1. Run K-Means on the training data to get *n* "prototype customers" (centres).
2. For each customer compute how close they are to each prototype: `exp(-gamma * distance^2)` - 1 means identical, about 0 means far.
3. Those *n* numbers are the hidden layer. Train a logistic regression on them (the output layer).
4. `gamma` = width of each neuron's neighbourhood; tuned on validation with `n_centers` and `C`.
In the notebook I showed that a very large gamma makes all activations about 0 (underfitting).

### Phase 7 - Recommender (`src/recommendations/`)
* **Popularity:** most distinct customers who bought a product.
* **Collaborative (item-based):** build a customer x product 0/1 matrix; cosine similarity between product columns ("bought by the same people"); score a customer's candidates by similarity to what they already bought; exclude owned products.
* **Content-based:** TF-IDF (1-2 grams) on product descriptions; cosine similarity to the customer's profile.
* **Hybrid:** `0.60 x collaborative + 0.30 x content + 0.10 x popularity`, each **min-max normalised** first (different scales would otherwise let one dominate). Weights come from the brief; I did not tune them on the test period.
* **Cold start:** unknown or thin-history customer -> popularity list, flagged `fallback`.
* **Explanations:** each item carries a reason ("Customers who bought X often buy this too").
* **Time-aware evaluation:** fit only on data before 2011-10-10; ground truth = *new* products bought afterwards (800 customers).

| | precision@5 | NDCG@5 | catalogue coverage |
|---|---|---|---|
| Popularity | 0.029 | 0.030 | 0.9% |
| Collaborative | 0.062 | 0.065 | 6.9% |
| Content | 0.060 | 0.069 | 19.3% |
| **Hybrid** | **0.066** | **0.072** | 10.6% |

Low absolute numbers are expected (predicting *new* purchases among 4,700 products). The meaningful result is hybrid about 2.3x popularity. Hybrid vs content-only is a small difference on one split - I would not claim it is significant.

### Phase 8 - Business insights (`src/analytics/business_insights.py`)
Every sentence is generated from results, not typed by hand. Examples: Champions carry 73% of revenue; At-Risk customers hold GBP 2.77M (17%) and have been silent a median 176 days; 28% of customers bought only once; November is the peak; the top product pair sells together 521 times (lift 31.8x).

### Phase 9 - Dashboard (`app.py`, `pages/`, `src/ui/`)
Six pages: Overview, Customer Segmentation, Customer Explorer, Product Recommendations, Model Performance, Business Insights. Plotly charts, teal/blue/indigo palette, amber reserved for "at-risk", light and dark themes, cached data loading. It only *reads* artefacts; it never trains.

### Phase 10 - API (`api/`)
`GET /health`, `GET /customers/{id}`, `GET /customers/{id}/segment`, `POST /customers/{id}/predict-value`, `POST /recommendations`, `GET /analytics/segments`.
Pydantic validation (422), 404 for unknown customers, optional API key (401), request-ID logging, Swagger at `/docs`. Responses contain derived metrics only - no invoices or purchase histories.

### Phase 11 - Tests and deployment
52 tests: cleaning rules on hand-made data with known answers, feature values computed by hand, the **no-future-leakage** test, segmentation sanity, RBF learning a non-linear boundary, protocol checks (selection uses validation only), recommender rules (never recommend owned items, normalisation, cold start), and a full pipeline -> API integration test. Deployment files: Dockerfile, `render.yaml`, Streamlit Cloud instructions (`docs/deployment.md`).

---

## 5. The decisions that matter (know the "why")

| Decision | Alternative | Why I chose it |
|---|---|---|
| Keep returns, net them against sales | Drop all negative quantities | Dropping them overstates revenue and hides unhappy customers |
| Frequency counts only sale invoices | Count all invoices | A return is not a purchase |
| Keep guest rows for company totals, drop for customer models | Drop them everywhere | They are 13.6% of revenue; customer features need an ID |
| Flag large wholesale lines, remove only absurd ones | Remove all outliers | Big buyers are real customers (and valuable) |
| log + scale before K-Means | Raw features | Money would dominate distance; spend is heavily skewed |
| Let silhouette choose k (got 4) | Force 6 segments | Brief said not to force it; 4 groups are what the data supports |
| Hungarian label matching | Hand-label clusters | Reproducible and unique labels |
| Predict *future* spend | Define "high value" from current spend | Label from current spend would be leakage and useless for action |
| Quantile label per window | Fixed GBP threshold | Christmas quarter would otherwise flood the positive class |
| Temporal train/val/test | Random split | Random split lets the model see the future; overstates performance |
| Tune and pick on validation, test once | Tune on test | Otherwise the test score is optimistic |
| PR-AUC + precision/recall floors | Accuracy | 80% negatives: "nobody is high value" is already 80% accurate |
| Evaluate recommendations on *new* products only | Include repeats | Repeat purchases are too easy and not a recommendation |
| Min-max normalise before blending | Blend raw scores | Raw scales differ, the biggest numbers would win silently |
| Commit processed data and models (~9 MB) | Ship code only | Dashboard and API run straight after cloning |

---

## 6. Results cheat sheet

* Data: 1,067,371 -> 1,021,241 rows; net revenue GBP 18.91M; return rate 2.43%.
* Customers: 5,852; 28% bought exactly once.
* Segments: 4 (silhouette 0.359); Champions 20% of customers / 73% of revenue.
* Value prediction (test): XGBoost ROC-AUC 0.868, PR-AUC 0.683, precision 0.55, recall 0.70; all models within about 0.01 AUC.
* Recommender: hybrid precision@5 0.066 vs popularity 0.029.
* Pipeline runtime about 1 minute; tests 52 passing; verified on Python 3.13 and 3.11.

---

## 7. Where to find things

| I want to explain... | Open this |
|---|---|
| How data is cleaned | `src/data/cleaning.py`, `docs/data_quality_report.md` |
| How features avoid leakage | `src/features/customer_features.py` (`history_before`), `tests/test_features.py` |
| Segmentation and labels | `src/models/customer_segmentation.py` (`choose_k`, `ARCHETYPES`) |
| The label and time windows | `src/models/value_prediction.py` (`make_windows`, `build_split`) |
| My RBF network | `src/models/rbf_network.py` |
| The recommender | `src/recommendations/hybrid.py`, `evaluation.py` |
| Everything wired together | `src/pipeline.py` |
| Constants (seed, horizon, colours) | `src/utils/config.py` |
| Dashboard theme | `src/ui/theme.py` |
| API | `api/main.py`, `api/routes/`, `api/schemas/models.py` |
| Honest metrics | `docs/model_evaluation_report.md` |
| Teaching version of everything | `notebooks/01...06` |

**Suggested reading order (about 2 hours):** README -> notebooks 01 to 06 -> `src/pipeline.py` -> then the module for any phase you want to defend.

---

## 8. A 5-minute demo script

1. **Overview page** (30s): "5,852 customers, GBP 18.9M net revenue; the donut shows 4 segments."
2. **Customer Segmentation** (60s): PCA plot - "groups touch, customers are a continuum"; revenue share chart - "Champions are 20% of customers, 73% of revenue".
3. **Customer Explorer** (45s): search a Champion (e.g. 12347), show RFM, percentile gauges, purchase history.
4. **Recommendations** (60s): same customer, hybrid list with component scores and reasons; mention fallback for new visitors.
5. **Model Performance** (60s): model table, ROC/PR, confusion matrix, SHAP/permutation importance; "models are close; protocol matters most".
6. **API** (30s): open `/docs`, call `POST /recommendations`.
7. Close with limitations (section 10).

---

## 9. Questions you should be able to answer

1. **Why not drop negative quantities?** They are returns; dropping them overstates revenue and loses a customer signal. I keep them as `return` records and use net revenue.
2. **What is data leakage and where could it have happened here?** Using information from after the prediction moment. Risks: features computed from all data, labels from current spend, scaler fitted on all data, tuning on test. I used a snapshot date with strict `<`, a future-only label, train-only preprocessing, and tests.
3. **How do you know there is no leakage?** `test_no_future_leakage`: features at a cutoff are identical after deleting later transactions.
4. **Why a temporal split?** In production we predict the future from the past; a random split mixes periods and inflates scores.
5. **Why PR-AUC and not accuracy?** The positive class is 20%; accuracy rewards predicting "no" for everyone. PR-AUC focuses on the rare class.
6. **Your models are all about equally good - why XGBoost?** Selected by a documented rule on validation (highest PR-AUC among models meeting precision/recall floors). Differences are within noise; I say so.
7. **Why did you get 4 clusters, not 6?** Silhouette preferred 4. Forcing 6 would split genuinely similar customers.
8. **Silhouette is only 0.36. Are the segments real?** They are a useful simplification of a continuum, stable across seeds (ARI 0.97+), and interpretable. I never claim natural clusters.
9. **How does K-Means work?** Pick k centres, assign each point to the nearest, move centres to the mean of their points, repeat. Inertia = total squared distance to centres.
10. **Why log-transform?** Spend and counts are right-skewed; logs make distance meaningful and stop the biggest buyers dominating.
11. **How does your RBF network work?** K-Means centres, Gaussian similarity to each centre as hidden units, logistic regression output layer; tune gamma, centres, C on validation.
12. **Why is test lower than validation?** Different period (Christmas quarter, distribution shift) and the threshold is fixed from validation. I report both.
13. **Why are recommender precision numbers so low?** I only count *new* products, among about 4,700, so hits are rare. The comparison against popularity is what matters.
14. **How does the hybrid handle a brand-new customer?** Falls back to popularity and flags it.
15. **How would you know recommendations really work?** An online A/B test; offline metrics on one split are noisy.
16. **How is customer privacy handled in the API?** Only derived metrics are returned - no invoice lines or histories; optional API key.
17. **What would you change with more time?** See section 10.

---

## 10. Limitations and next steps (say these before someone else does)

* Only two years of data: one Christmas per year, so seasonality and growth cannot be separated.
* The test window is the Christmas quarter, a deliberate distribution shift; one test split only (no repeated rolling evaluation).
* The final model is trained on the training period only (not refit on train + validation) to keep evaluation clean.
* Recommender: 800 evaluation customers, one split, no significance tests, weights not tuned.
* Guest checkouts (13.6% of revenue) are invisible to customer models.
* The committed results were produced from a GitHub copy of Online Retail II whose size and date range match UCI (not compared byte for byte).
* Not done here: cloud deployment and Docker image build (files are provided).
* Next steps: rolling-window backtests, calibrated probabilities and cost-based thresholds, tuning hybrid weights on a validation period, online A/B test, drift monitoring, scheduled retraining.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Observation / feature | A row / a column |
| EDA | Looking at data before modelling |
| RFM | Recency, Frequency, Monetary |
| Snapshot date | The "today" for which features are computed |
| Leakage | Using future information when predicting |
| Skew | Long tail on one side (money and counts are right-skewed) |
| K-Means / centroid / inertia | Clustering algorithm / cluster centre / total squared distance to centres |
| Silhouette | -1..1 score of how well separated clusters are |
| PCA | Squash many dimensions to 2 for plotting |
| Precision / recall / F1 | Share of flagged that are right / share of true found / their balance |
| ROC-AUC / PR-AUC | Ranking quality overall / on the rare class |
| Overfitting / underfitting | Memorising noise / too simple |
| RBF network | Hidden units = Gaussian closeness to prototype points, linear output layer |
| SHAP / permutation importance | Per-prediction feature contributions / score drop when a feature is shuffled |
| Cosine similarity | Angle-based similarity of two vectors |
| TF-IDF | Word weight = frequent here, rare elsewhere |
| Cold start | No history for a user/item |
| Coverage / popularity bias | Share of catalogue ever recommended / tendency to push best-sellers |
| NDCG@k | Ranking score that rewards hits near the top |

---

## 12. Check yourself (answers are in this guide)

1. Name three data-quality problems and how each was handled.
2. Draw the three time windows and say what is allowed to see what.
3. Why is the label a quantile per window?
4. Explain the Hungarian step in one sentence.
5. What does gamma do in the RBF network?
6. Why normalise before blending recommender scores?
7. Which result would you be careful *not* to over-claim, and why?
