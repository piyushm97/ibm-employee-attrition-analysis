
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
from pathlib import Path

from src.preprocess import TARGET_COL, prepare_features, split_X_y

st.set_page_config(page_title="IBM Employee Attrition Analysis", layout="wide")

st.title("IBM Employee Attrition Analysis")
st.caption("Upload a CSV or use the sample to explore, train a model, and score attrition risk.")

# --- Data loader
uploaded = st.file_uploader("Upload CSV (must include Attrition column as Yes/No)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("Loaded uploaded dataset.")
else:
    df = pd.read_csv("data/sample_ibm_attrition.csv")
    st.info("Using sample dataset bundled with the app.")

st.write("### Preview")
st.dataframe(df.head())

# --- EDA
with st.expander("Exploratory Data Analysis", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.write("Attrition Counts")
        cnt = df[TARGET_COL].value_counts()
        fig, ax = plt.subplots()
        cnt.plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.write("Monthly Income by Attrition")
        if "MonthlyIncome" in df.columns:
            fig2, ax2 = plt.subplots()
            sns.boxplot(data=df, x=TARGET_COL, y="MonthlyIncome", ax=ax2)
            st.pyplot(fig2)

# --- Model training
st.write("## Train Model")
algo = st.selectbox("Algorithm", ["Random Forest", "Logistic Regression"])
train_button = st.button("Train on Current Data")

model_path = Path("models/attrition_model.joblib")
if train_button:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X, y = split_X_y(df)
    X_enc = prepare_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=y)

    if algo == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    st.success(f"Model trained. Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_test, y_prob)
            st.info(f"ROC AUC={auc:.3f}")
        except Exception:
            pass

    # Save model + the columns used for encoding for future scoring
    dump({"model": clf, "columns": list(X_enc.columns)}, model_path)
    st.write(f"Saved model to `{model_path}`")

# --- Scoring
st.write("## Score New Data")
st.caption("Drop a dataset with the same columns (without Attrition), or we will ignore the Attrition column if present.")
score_file = st.file_uploader("Upload CSV to score (optional)", type=["csv"], key="score")
if score_file is not None:
    score_df = pd.read_csv(score_file)
else:
    score_df = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df.copy()

if model_path.exists():
    bundle = load(model_path)
    model = bundle["model"]
    cols = bundle["columns"]
    X_new = prepare_features(score_df)
    # align columns
    for c in cols:
        if c not in X_new.columns:
            X_new[c] = 0
    X_new = X_new[cols]
    preds = model.predict(X_new)
    probs = model.predict_proba(X_new)[:,1] if hasattr(model, "predict_proba") else None

    out = score_df.copy()
    out["Attrition_Pred"] = preds
    if probs is not None:
        out["Attrition_Prob"] = probs

    st.write("### Predictions")
    st.dataframe(out.head())
    st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"), file_name="attrition_predictions.csv", mime="text/csv")
else:
    st.warning("Train a model first to enable scoring.")
