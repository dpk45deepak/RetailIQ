"""RetailIQ dashboard entry point.   Run:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="RetailIQ", page_icon=":material/insights:", layout="wide", initial_sidebar_state="expanded")

from src.ui import data, theme  # noqa: E402

theme.setup()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")
data.ensure_artifacts()

pages = st.navigation({
    "Analytics": [
        st.Page("pages/1_overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("pages/2_customer_segmentation.py", title="Customer Segmentation", icon=":material/hub:"),
        st.Page("pages/3_customer_explorer.py", title="Customer Explorer", icon=":material/person_search:"),
    ],
    "Intelligence": [
        st.Page("pages/4_product_recommendations.py", title="Product Recommendations", icon=":material/shopping_bag:"),
        st.Page("pages/5_model_performance.py", title="Model Performance", icon=":material/query_stats:"),
        st.Page("pages/6_business_insights.py", title="Business Insights", icon=":material/lightbulb:"),
    ],
})

meta = data.load_json("run_metadata.json")
with st.sidebar:
    kind = "Demo data (synthetic)" if meta.get("is_synthetic") else "UCI Online Retail II"
    theme.html(f'<div class="rq-side-foot"><b>{kind}</b><br>{meta["customers"]:,} customers · {meta["rows_clean"]:,} transaction lines<br>'
               f'Built {meta["built_at"][:10]}</div>')

pages.run()
