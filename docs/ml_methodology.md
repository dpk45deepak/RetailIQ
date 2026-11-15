# ML methodology

## 1. Segmentation (unsupervised)
* **Features:** recency, frequency, monetary, tenure. `log1p` (negative net spend clipped at 0) then `StandardScaler`.
* **Choosing k:** evaluate k = 2..10 (inertia and silhouette). Among k = 3..6 pick the best silhouette; if a larger k is within 0.02, prefer it. Six is the cap only because six business labels exist - **the data decides**, and it chose **4 segments** (silhouette 0.359).
* **Labels:** each cluster's median R/F/M/tenure percentile profile is matched to archetype profiles (Champions, Loyal, Potential Loyalists, New, At-Risk, Hibernating) with the Hungarian algorithm, so labels are unique and not hand-picked. Segments found: Champions, At-Risk, Potential Loyalists, Hibernating.
* **Stability:** adjusted Rand index against 5 other seeds is 0.97-1.00. PCA (2 components, 94.6% of variance) is used for plotting only.
* **Caveat:** silhouette ~0.36 = moderately separated groups; customers form a continuum.

## 2. Value prediction (supervised)
* **Task:** predict whether a customer's **net revenue in the next 90 days is in the top 20%** of known customers (quantile computed per window so Christmas and summer are comparable). Customers who do not return have revenue 0.
* **Temporal protocol** (90-day horizon): train cut-off 2011-03-15, validation 2011-06-13, test 2011-09-11, data ends 2011-12-10. Features use only data before each cut-off; labels only the next 90 days; each label window ends before the next cut-off.
* **Preprocessing pipeline** (fitted on train only): median imputation (+ missing indicators), signed log1p on skewed money/count features, standard scaling.
* **Models:** logistic regression, random forest, XGBoost, **RBF network** (K-Means centres -> Gaussian activations -> L2 logistic output layer; `gamma`, `C`, centres tuned on validation), MLP.
* **Selection:** hyper-parameters and the decision threshold (max F1) are chosen on validation. Winner = highest validation PR-AUC among models meeting minimum precision (0.4) and recall (0.5). Test is scored once afterwards. Results: [`model_evaluation_report.md`](model_evaluation_report.md).
* **Explainability:** permutation importance (selected model) and SHAP (XGBoost).
* **Known limitation:** the test window is the Christmas quarter, a deliberate distribution shift; also, the final model is trained on the training period only (not refit on train+validation), trading a little data for a clean evaluation.

## 3. Recommendations
* **Interaction matrix:** customers x products (binary purchase) from sale lines before the evaluation split.
* **Item-based collaborative filtering:** cosine similarity of product columns; score = summed similarity to already-bought items; already-bought items excluded.
* **Content-based:** TF-IDF (1-2 grams) on normalised descriptions; cosine similarity to the customer's profile.
* **Hybrid:** `0.60 collaborative + 0.30 content + 0.10 popularity`, each min-max normalised over candidates first; popularity fallback for unknown or thin-history customers.
* **Evaluation:** fit on data before 2011-10-10; ground truth = *new* products bought afterwards; 800 sampled customers; Precision/Recall/NDCG@5, catalogue coverage, popularity bias.
* **Limitation:** one offline split with 800 customers is noisy. Hybrid vs content-only differences are small and not statistically tested; confirm with an A/B test.

## 4. Anti-leakage checklist (enforced by tests)
1. Features at cut-off T identical with or without data after T (`test_no_future_leakage`).
2. Label not derived from any input feature.
3. Imputer/scaler fitted on train only.
4. Test never used for tuning, threshold or model choice (`test_experiment_respects_the_protocol`).
5. Recommenders fitted only on data before the split date.
