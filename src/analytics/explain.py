"""Model explainability: which features drive the high-value prediction?

Two complementary tools
-----------------------
* PERMUTATION IMPORTANCE (model-agnostic): shuffle one feature on held-out data and measure how much the score drops.
  A big drop = the model relied on that feature.
* SHAP (for tree models): splits each individual prediction into per-feature contributions. Mean |SHAP| is a global
  importance; the correlation between a feature's value and its SHAP value tells the DIRECTION of the effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.utils.config import RANDOM_STATE, get_logger

log = get_logger(__name__)


def permutation_table(pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance of the FULL pipeline on held-out data (average precision)."""
    r = permutation_importance(pipeline, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=RANDOM_STATE)
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def shap_table(xgb_model, prep, X: pd.DataFrame, max_rows: int = 600) -> pd.DataFrame | None:
    """Mean |SHAP| and effect direction per ORIGINAL feature. Returns None if SHAP is unavailable."""
    try:
        import shap
        Xs = X.sample(min(max_rows, len(X)), random_state=RANDOM_STATE)
        Z = prep.transform(Xs)
        values = shap.TreeExplainer(xgb_model).shap_values(Z)
        values = np.asarray(values)
        n = len(X.columns)                       # imputer may append missing-indicator columns at the end; ignore them
        values, Z = values[:, :n], Z[:, :n]
        direction = [float(np.corrcoef(Z[:, j], values[:, j])[0, 1]) if values[:, j].std() > 0 else 0.0 for j in range(n)]
        return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(0), "direction_corr": direction})
                .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    except Exception as exc:                      # never let an optional explainer break the pipeline
        log.warning("SHAP unavailable (%s); falling back to permutation importance only.", exc)
        return None
