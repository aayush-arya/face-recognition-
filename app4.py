import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------
# 1. Load dataset
# -------------------------
st.title("Face Recognition Classification App")

file_path = "Face Recognition Image.xlsx"

try:
    df = pd.read_excel(file_path)
    st.success("✅ Dataset loaded successfully")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Handle missing values & encode
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

for col in df.select_dtypes(include=["object"]).columns:
    if col != "file":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Extract labels from filenames
if "file" in df.columns:
    df["label"] = df["file"].apply(lambda x: str(x).split("_")[0])
else:
    st.error("⚠️ No 'file' column found in dataset")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df.drop(["file", "label"], axis=1)
y = df["label"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
stratify_option = y if y.value_counts().min() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_option
)

# -------------------------
# 2. Train Baseline Model (Logistic Regression)
# -------------------------
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

st.subheader("Baseline Model - Logistic Regression")
st.text(classification_report(y_test, y_pred_base))

# -------------------------
# 3. Train Improved Model (Random Forest tuned)
# -------------------------
rf = RandomForestClassifier(
    n_estimators=200, max_depth=20,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

st.subheader("Improved Model - Random Forest")
st.text(classification_report(y_test, y_pred_rf))

# -------------------------
# 4. Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred_rf)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf.classes_, yticklabels=rf.classes_, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Random Forest")
st.pyplot(fig)

# -------------------------
# 5. ROC Curve (only works if binary classification)
# -------------------------
if len(np.unique(y)) == 2:
    y_prob = rf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=rf.classes_[1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0,1], [0,1], 'r--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=rf.classes_[1])
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(recall, precision, label="Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    st.pyplot(fig)

# -------------------------
# 6. Compare Baseline vs Improved
# -------------------------
st.subheader("Model Comparison")
st.write("🔹 Baseline Accuracy:", baseline.score(X_test, y_test))
st.write("🔹 Improved Model Accuracy:", rf.score(X_test, y_test))

