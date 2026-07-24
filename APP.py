import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score

# Page Configuration
st.set_page_config(
    page_title="CRASH Gambling Risk Engine", layout="wide"
)

st.title(" CRASH Gambling Risk & Addiction Detection Dashboard")
st.markdown("---")


#  Data & Model Loader
@st.cache_resource
def load_assets():
    model_path = "xgb_gambling_risk_model.pkl"
    x_test_path = "X_test.npy"
    y_test_path = "y_test.npy"

    if (
        os.path.exists(model_path)
        and os.path.exists(x_test_path)
        and os.path.exists(y_test_path)
    ):
        model = joblib.load(model_path)
        X_test = np.load(x_test_path, allow_pickle=True)
        y_test = np.load(y_test_path, allow_pickle=True)
        y_scores = model.predict_proba(X_test)[:, 1]
        is_mock = False
    else:
        # Graceful fallback for demonstration when raw model files are absent
        np.random.seed(42)
        y_test = np.random.choice([0, 1], size=6421, p=[0.95, 0.05])
        y_scores = np.where(
            y_test == 1,
            np.random.uniform(0.1, 0.95, size=6421),
            np.random.uniform(0.0, 0.45, size=6421),
        )
        X_test = np.random.uniform(0, 5, size=(6421, 3))
        is_mock = True

    return y_test, y_scores, X_test, is_mock


y_test, y_scores, X_test, is_mock = load_assets()

if is_mock:
    st.info(
        " **Demo Mode Active:** Model artifact (`xgb_gambling_risk_model.pkl`) not detected locally. Displaying simulated probability distributions."
    )

#  Sidebar Controls
st.sidebar.header(" Operational Controls")
st.sidebar.markdown(
    "Adjust the decision threshold to balance false positive compliance load against target recall."
)

threshold_pct = st.sidebar.slider(
    label="Suspicion Certainty Threshold (%)",
    min_value=5,
    max_value=90,
    value=30,
    step=5,
)
threshold = threshold_pct / 100.0

# Dynamic Calculations
y_pred = (y_scores >= threshold).astype(int)

total_users = len(y_test)
flagged_users = int(np.sum(y_pred == 1))
true_positives = int(np.sum((y_pred == 1) & (y_test == 1)))
false_positives = int(np.sum((y_pred == 1) & (y_test == 0)))

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

#  Dashboard Header Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(" Total Accounts Screened", f"{total_users:,}")
with col2:
    st.metric(
        " Flagged for Review",
        f"{flagged_users:,}",
        delta=f"{flagged_users/total_users*100:.1f}% of total",
    )
with col3:
    st.metric(" Precision", f"{precision*100:.1f}%")
with col4:
    st.metric(" Target Recall", f"{recall*100:.1f}%")

st.markdown("---")

# 6. Operational Breakdown & Risk Stance
st.subheader("Compliance Triage Queue")

left_col, right_col = st.columns(2)

with left_col:
    st.markdown(f"""
    ### Current Triage Summary
    * **At-Risk Accounts Captured:** **{true_positives}**
    * **False Positive Reviews:** **{false_positives}**
    * **Total Human Review Queue:** **{flagged_users}** accounts
    """)

with right_col:
    if threshold_pct <= 20:
        st.warning(
            " **Strategy: Maximum Recall (High Sensitivity)**\n"
            "Captures nearly all at-risk users. Compliance queue will receive a high volume of false positives needing manual review."
        )
    elif threshold_pct >= 70:
        st.success(
            " **Strategy: High Precision (Resource Constrained)**\n"
            "Minimizes false alarms for compliance staff. Flags only accounts demonstrating high-certainty addictive signatures."
        )
    else:
        st.info(
            " **Strategy: Balanced Triage**\n"
            "Maintains steady intervention coverage while keeping false positive review volume manageable."
        )

# 7. Sample Flagged Queue Data Table
st.markdown("---")
st.subheader(" Flagged Accounts Preview")

if flagged_users > 0:
    flagged_indices = np.where(y_pred == 1)[0]
    sample_size = min(10, len(flagged_indices))
    sample_ids = flagged_indices[:sample_size]

    df_preview = pd.DataFrame(
        {
            "Account_Index": sample_ids,
            "Model_Suspicion_Score": np.round(y_scores[sample_ids], 3),
            "Wager_Volatility": np.round(X_test[sample_ids, 0], 2),
            "Loss_Chase_Ratio": np.round(X_test[sample_ids, 1], 2),
            "Scalper_Ratio": np.round(X_test[sample_ids, 2], 2),
            "Status": ["Flagged for Review"] * sample_size,
        }
    )

    st.dataframe(df_preview, use_container_width=True)
else:
    st.write("No accounts flagged at the current threshold level.")
