"""Model Performance: compare the five classifiers on held-out data and see what drives predictions."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import data, theme

vm, meta = data.load_json("value_metrics.json"), data.load_json("run_metadata.json")
models, sel = vm["models"], vm["selected_model"]
w = vm["windows"]

theme.hero("Model performance",
           f"Predicting who will be a top-20% customer in the next {w['horizon_days']} days. Models are trained on the past and judged on a later, untouched period.",
           theme.meta_chips(meta) + [(f"Selected: {sel}", False)])

theme.html(f'<div class="rq-card" style="margin-bottom:6px"><b>How it was tested.</b> Features come only from before each cutoff; labels come from the following {w["horizon_days"]} days. '
           f'Train cutoff {w["train"][:10]}, validation {w["val"][:10]}, test {w["test"][:10]}. '
           f'Hyper-parameters, the decision threshold and the winning model were all chosen on validation data. The test period was scored once.<br>'
           f'<span class="rq-note">Selection rule: {theme.esc(vm["selection_rule"])}.</span></div>')

split = st.segmented_control("Evaluate on", ["Test (untouched)", "Validation"], default="Test (untouched)", label_visibility="collapsed") or "Test (untouched)"
key = "test" if split.startswith("Test") else "validation"
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
nice = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}

theme.section("Model comparison", "Selected model is marked. Threshold is tuned for best F1 on validation data.")
tbl = pd.DataFrame([{"Model": ("★ " if n == sel else "") + n, **{nice[m]: r[key][m] for m in metrics}, "CV ROC-AUC (train)": r["cv_roc_auc_train_mean"]} for n, r in models.items()])
cfg = {nice[m]: st.column_config.ProgressColumn(nice[m], format="%.3f", min_value=0, max_value=1, width="small") for m in metrics}
cfg["CV ROC-AUC (train)"] = st.column_config.NumberColumn("5-fold CV AUC (train)", format="%.3f", help="Cross-validation on the training period only.")
theme.show_df(tbl, column_config=cfg)

pick = st.selectbox("Inspect model", list(models), index=list(models).index(sel))
r = models[pick][key]
theme.kpis([{"label": nice[m], "value": f"{r[m]:.3f}", "accent": theme.ACCENTS[i % 3]} for i, m in enumerate(["roc_auc", "pr_auc", "precision", "recall", "f1"])])

c1, c2 = st.columns(2, gap="large")
with c1:
    theme.section("Confusion matrix", f"{pick} at threshold {r['threshold']:.2f}.")
    cm = np.array(r["confusion_matrix"])
    tot = cm.sum()
    lab = np.array([["True negative", "False positive"], ["False negative", "True positive"]])
    text = [[f"<b>{cm[i, j]:,}</b><br>{lab[i, j]}<br>{cm[i, j] / tot:.0%}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=cm / cm.sum(axis=1, keepdims=True), x=["Predicted: not high value", "Predicted: high value"], y=["Actual: not high value", "Actual: high value"],
                               text=text, texttemplate="%{text}", colorscale=[[0, "#e9f1fd"], [1, "#184f95"]], showscale=False, xgap=4, ygap=4, zmin=0, zmax=1,
                               hovertemplate="%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    theme.show(theme.style(fig, 340, legend=False))
    st.caption("Shading shows the share of each actual class (rows sum to 100%).")
with c2:
    theme.section("ROC curve", "Higher and further left is better. The diagonal is random guessing.")
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="rgba(120,120,120,.7)"), name="Random", hoverinfo="skip")
    for n, m in models.items():
        c = m[key]["roc_curve"]
        on = n == pick
        fig.add_scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=n, line=dict(width=3.2 if on else 1.4, color=theme.BLUE if on else None),
                        opacity=1 if on else 0.55, hovertemplate=n + "<br>FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>")
    fig.update_xaxes(title="False positive rate", range=[0, 1], showgrid=True)
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    theme.show(theme.style(fig, 340))

c3, c4 = st.columns(2, gap="large")
with c3:
    theme.section("Precision-recall curve", "Most useful when high-value customers are the minority (about 1 in 5).")
    fig = go.Figure()
    for n, m in models.items():
        c = m[key]["pr_curve"]
        on = n == pick
        fig.add_scatter(x=c["recall"], y=c["precision"], mode="lines", name=n, line=dict(width=3.2 if on else 1.4, color=theme.INDIGO if on else None),
                        opacity=1 if on else 0.55, hovertemplate=n + "<br>Recall %{x:.2f} · Precision %{y:.2f}<extra></extra>")
    fig.update_xaxes(title="Recall", range=[0, 1], showgrid=True)
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    theme.show(theme.style(fig, 340))
with c4:
    theme.section("What drives predictions", "Larger = the model relies on it more.")
    ex = vm["explainability"]
    tabs = st.tabs(["SHAP (XGBoost)", f"Permutation ({sel})"] if ex.get("shap") else [f"Permutation ({sel})"])
    if ex.get("shap"):
        with tabs[0]:
            s = pd.DataFrame(ex["shap"]).head(10).iloc[::-1]
            fig = go.Figure(go.Bar(y=s["feature"].str.replace("_", " "), x=s["mean_abs_shap"], orientation="h",
                                   marker_color=[("#eb6834" if d < -0.15 else theme.TEAL) for d in s["direction_corr"]],
                                   customdata=s["direction_corr"], hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<br>value-effect correlation %{customdata:+.2f}<extra></extra>"))
            fig.update_xaxes(title="Mean |SHAP value|", showgrid=True)
            fig.update_yaxes(showgrid=False)
            theme.show(theme.style(fig, 300, legend=False))
            st.caption("Teal: higher values push toward high-value. Orange: higher values push away (e.g. long inactivity).")
    with tabs[-1]:
        p = pd.DataFrame(ex["permutation"]).head(10).iloc[::-1]
        fig = go.Figure(go.Bar(y=p["feature"].str.replace("_", " "), x=p["importance"], orientation="h", marker_color=theme.INDIGO,
                               error_x=dict(type="data", array=p["std"], color="rgba(120,120,120,.6)"), hovertemplate="%{y}<br>drop in PR-AUC %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="Drop in PR-AUC when shuffled", showgrid=True)
        fig.update_yaxes(showgrid=False)
        theme.show(theme.style(fig, 300, legend=False))

with st.expander("Details: best hyper-parameters, label rule and split sizes"):
    st.markdown(f"**Label:** {vm['label_rule']}. Positive rate: train {vm['positive_rate']['train']:.1%}, validation {vm['positive_rate']['val']:.1%}, test {vm['positive_rate']['test']:.1%}. "
                f"Customers: {vm['sizes']['train']:,} / {vm['sizes']['val']:,} / {vm['sizes']['test']:,}.")
    theme.show_df(pd.DataFrame([{"Model": n, "Best parameters (chosen on validation)": ", ".join(f"{k}={v}" for k, v in m["best_params"].items())} for n, m in models.items()]))
    st.caption("All scores come from the pipeline run recorded in models/value_metrics.json. They describe this dataset only and are not guarantees about future performance.")
