# streamlit_app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

warnings.filterwarnings("ignore")

import streamlit as st

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

# ------- Load data (cached) -------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

data_path = st.sidebar.text_input("CSV file path", "credit card fraud.csv")
if not data_path:
    st.sidebar.error("Please provide the CSV file path (eg. credit card fraud.csv)")
    st.stop()

try:
    df = load_data(data_path)
except Exception as e:
    st.sidebar.error(f"Error loading file: {e}")
    st.stop()

# sample option
sample_frac = st.sidebar.slider("Sample fraction (for faster runs)", 0.05, 1.0, 0.1, 0.05)

if sample_frac < 1.0:
    df = df.sample(frac=sample_frac, random_state=48)

# ------- Sidebar: show data/info -------
st.sidebar.subheader("Data preview")
if st.sidebar.checkbox("Show dataframe (first 100 rows)"):
    st.write(df.head(100))
    st.write("Shape:", df.shape)
    st.write(df.describe())

# show fraud / valid counts
if "Class" not in df.columns:
    st.error("Dataset must contain 'Class' column (0 = valid, 1 = fraud).")
    st.stop()

fraud = df[df.Class == 1]
valid = df[df.Class == 0]
outlier_percentage = (len(fraud) / len(valid)) * 100 if len(valid) > 0 else 0

st.sidebar.markdown(f"**Fraudulent transactions:** {len(fraud)}")
st.sidebar.markdown(f"**Valid transactions:** {len(valid)}")
st.sidebar.markdown(f"**Outlier %:** {outlier_percentage:.3f}%")

# ------- Prepare features and labels -------
X = df.drop(columns=["Class"])
y = df["Class"]

# allow user to pick features (or use all)
st.sidebar.subheader("Features")
use_all = st.sidebar.checkbox("Use all features (recommended)", value=True)
if not use_all:
    cols = st.sidebar.multiselect("Select features to use", list(X.columns), default=list(X.columns)[:10])
    if not cols:
        st.sidebar.error("Select at least one feature")
        st.stop()
    X = X[cols]

# split
test_size = st.sidebar.slider("Test set size", 0.2, 0.4, 0.25, 0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

if st.sidebar.checkbox("Show train/test shapes"):
    st.write("X_train:", X_train.shape, "y_train:", y_train.shape)
    st.write("X_test:", X_test.shape, "y_test:", y_test.shape)

# ------- Model & imbalance options -------
st.sidebar.subheader("Model & Imbalance handling")
model_name = st.sidebar.selectbox("Choose classifier",
                                  ["Logistic Regression", "KNN", "Random Forest", "Extra Trees"])
imbalance = st.sidebar.selectbox("Imbalance handling", ["None", "SMOTE (oversample)", "NearMiss (undersample)"])

# classifier mapping
def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    elif name == "KNN":
        return KNeighborsClassifier()
    elif name == "Random Forest":
        return RandomForestClassifier(random_state=42, n_estimators=100)
    elif name == "Extra Trees":
        return ExtraTreesClassifier(random_state=42, n_estimators=100)
    else:
        return LogisticRegression(max_iter=1000)

model = get_model(model_name)

# feature importance helper
def get_feature_importances(model, X_train, y_train):
    has_attr = hasattr(model, "feature_importances_")
    if not has_attr:
        return None
    model.fit(X_train, y_train)
    return pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# ------- Performance computation -------
def run_training(model, X_train, y_train, X_test, y_test, imbalance):
    start = time.time()
    X_tr, y_tr = X_train.copy(), y_train.copy()

    if imbalance == "SMOTE (oversample)":
        sm = SMOTE(random_state=42)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    elif imbalance == "NearMiss (undersample)":
        nm = NearMiss()
        X_tr, y_tr = nm.fit_resample(X_tr, y_tr)

    # cross-val score (accuracy)
    try:
        cv_score = cross_val_score(model, X_tr, y_tr, cv=3, scoring="accuracy").mean()
    except Exception:
        cv_score = None

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    elapsed = time.time() - start
    return {"model": model, "cv_score": cv_score, "accuracy": acc, "confusion_matrix": cm, "class_report": cr, "time": elapsed}

# ------- UI: Run button -------
st.subheader("Run Model")
if st.button("Run training & evaluate"):
    with st.spinner("Training and evaluating..."):
        result = run_training(model, X_train, y_train, X_test, y_test, imbalance)

    st.success(f"Done — time: {result['time']:.2f}s")
    if result["cv_score"] is not None:
        st.write(f"Cross-val accuracy (3-fold): **{result['cv_score']:.4f}**")
    st.write(f"Test accuracy: **{result['accuracy']:.4f}**")

    # confusion matrix plot
    cm = result["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # classification report table
    st.write("### Classification Report")
    cr_df = pd.DataFrame(result["class_report"]).transpose()
    st.dataframe(cr_df)

    # feature importance if available
    fi = get_feature_importances(get_model(model_name), X_train, y_train)
    if fi is not None:
        st.write("### Feature Importances")
        st.bar_chart(fi.head(20))
    else:
        st.info("Selected model does not provide feature importances. Try Random Forest or Extra Trees to view importances.")

    st.balloons()
else:
    st.info("Configure options from the sidebar and click 'Run training & evaluate' to execute models.")
