import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
import importlib.util

# Page configuration
st.set_page_config(
    page_title="Bank Marketing ML Dashboard",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).parent

st.title("Bank Marketing Classification Dashboard")
st.subheader("M.Tech AIML | Machine Learning Assignment 2")

st.write(
    "Evaluate and compare five trained classification models "
    "using the Bank Marketing dataset."
)

# Load models
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

@st.cache_resource
def load_models():
    loaded = {}

    for name, filename in MODEL_FILES.items():
        path = BASE_DIR / "model" / filename
        loaded[name] = joblib.load(path)

    return loaded

@st.cache_data
def load_features():
    with open(BASE_DIR / "feature_columns.json") as f:
        return json.load(f)

try:
    models = load_models()
    feature_columns = load_features()

except Exception as e:
    st.error(f"Unable to load saved models: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your test CSV file to begin.")
    st.stop()

# Read dataset
try:
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("Uploaded CSV is empty.")
        st.stop()

    if "y" not in df.columns:
        st.error("Target column 'y' is missing.")
        st.stop()

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        st.error(f"Missing features: {missing}")
        st.stop()

    X_test = df[feature_columns].copy()

    y_test = (
        df["y"].astype(str).str.strip().str.lower()
    )

    if not y_test.isin(["yes", "no"]).all():
        st.error("Target column must contain yes/no values.")
        st.stop()

    if y_test.nunique() != 2:
        st.error("Both target classes are required for AUC.")
        st.stop()

except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()

# Dataset overview
st.header("1. Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Records", len(df))
c2.metric("Input Features", len(feature_columns))
c3.metric("Target Classes", y_test.nunique())

with st.expander("View Dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# Evaluate all models
st.header("2. Model Performance Comparison")

results = []
predictions = {}

y_binary = (y_test == "yes").astype(int)

try:
    for name, model in models.items():

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)

        predictions[name] = y_pred

        results.append({
            "ML Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_binary, scores),
            "Precision": precision_score(
                y_test, y_pred, pos_label="yes",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, pos_label="yes",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, y_pred, pos_label="yes",
                zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

except Exception as e:
    st.error(f"Evaluation failed: {e}")
    st.stop()

results_df = pd.DataFrame(results)

st.dataframe(
    results_df.style.format({
        col: "{:.4f}" for col in results_df.columns
        if col != "ML Model"
    }),
    use_container_width=True
)

# Select model
st.header("3. Individual Model Evaluation")

selected = st.selectbox(
    "Select Model",
    list(models.keys())
)

pred = predictions[selected]

selected_metrics = results_df[
    results_df["ML Model"] == selected
].iloc[0]

# Six metrics
st.subheader("Evaluation Metrics")

metric_names = [
    "Accuracy", "AUC", "Precision",
    "Recall", "F1 Score", "MCC"
]

columns = st.columns(3)

for i, metric in enumerate(metric_names):
    columns[i % 3].metric(
        metric,
        f"{selected_metrics[metric]:.4f}"
    )

# Confusion matrix
st.header("4. Confusion Matrix")

cm = confusion_matrix(
    y_test,
    pred,
    labels=["no", "yes"]
)

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(selected)

st.pyplot(fig)
plt.close(fig)

# Classification report
st.header("5. Classification Report")

report = classification_report(
    y_test,
    pred,
    labels=["no", "yes"],
    target_names=["No Subscription", "Subscription"],
    output_dict=True,
    zero_division=0
)

st.dataframe(
    pd.DataFrame(report).transpose(),
    use_container_width=True
)

# Comparison chart
st.header("6. Model Comparison Chart")

metric = st.selectbox(
    "Choose comparison metric",
    ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="ML Model",
    y=metric,
    ax=ax
)

ax.set_title(f"Model Comparison - {metric}")
plt.xticks(rotation=20)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Download results
st.header("7. Download Results")

st.download_button(
    "Download Model Results CSV",
    results_df.to_csv(index=False),
    file_name="model_comparison_results.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "All models are loaded from saved Joblib files. "
    "No model retraining is performed in this application."
)

st.write("Python version:", sys.version)
st.write("Python executable:", sys.executable)
st.write("Joblib location:", importlib.util.find_spec("joblib"))
