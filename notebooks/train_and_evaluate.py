"""
train_and_evaluate.py
-----------------------
Predictive Modeling Using Machine Learning

Steps:
1. Load dataset, encode categoricals, split train/test
2. Train Logistic/Linear Regression, Decision Tree, and Random Forest classifiers
3. Evaluate accuracy, precision, recall, F1
4. Visualize: confusion matrices, ROC curves, feature importance
5. Save a summary report (used to populate the README)
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

BASE = "/home/claude/predictive-modeling-ml-project"
DATA_PATH = f"{BASE}/data/customer_churn.csv"
VISUALS_DIR = f"{BASE}/visuals"
OUTPUT_DIR = f"{BASE}/output"
REPORT_PATH = f"{OUTPUT_DIR}/summary_report.json"

os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

report = {}

# ---------------------------------------------------------
# 1. LOAD & PREPARE
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
report["total_rows"] = len(df)
report["churn_rate"] = round(float(df["Churn"].mean()), 4)

le = LabelEncoder()
df["ContractType_enc"] = le.fit_transform(df["ContractType"])

feature_cols = [
    "Age", "TenureMonths", "MonthlyCharge", "SupportCalls",
    "ContractType_enc", "HasInternetService", "SatisfactionScore",
]
X = df[feature_cols]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

report["train_rows"] = len(X_train)
report["test_rows"] = len(X_test)

# ---------------------------------------------------------
# 2. TRAIN MODELS
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42),
}

results = {}
roc_data = {}
cleaned_metrics_table = []

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
    }
    roc_data[name] = (fpr, tpr, roc_auc)
    cleaned_metrics_table.append({
        "Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
        "Recall": round(rec, 4), "F1": round(f1, 4), "ROC_AUC": round(roc_auc, 4),
    })

report["model_results"] = results
best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
report["best_model"] = best_model_name

# ---------------------------------------------------------
# 3. VISUALIZATIONS
# ---------------------------------------------------------

# Chart 1: Model comparison bar chart (accuracy, precision, recall, f1)
metrics_df = pd.DataFrame(cleaned_metrics_table).set_index("Model")
fig, ax = plt.subplots(figsize=(8, 5))
metrics_df[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/01_model_comparison.png")
plt.close()

# Chart 2: Confusion matrices (3 subplots)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (name, res) in zip(axes, results.items()):
    cm = np.array(res["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/02_confusion_matrices.png")
plt.close()

# Chart 3: ROC curves
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Comparison")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/03_roc_curves.png")
plt.close()

# Chart 4: Feature importance (Random Forest)
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4.5))
importances.plot(kind="barh", ax=ax, color="#2a9d8f")
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/04_feature_importance.png")
plt.close()

# Chart 5: Churn distribution by contract type (exploratory)
fig, ax = plt.subplots(figsize=(7, 4.5))
churn_by_contract = df.groupby("ContractType")["Churn"].mean().sort_values(ascending=False)
sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values, ax=ax, hue=churn_by_contract.index,
            palette="rocket", legend=False)
ax.set_title("Churn Rate by Contract Type")
ax.set_ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/05_churn_by_contract_type.png")
plt.close()

report["feature_importance"] = importances.sort_values(ascending=False).round(4).to_dict()
report["churn_by_contract_type"] = churn_by_contract.round(4).to_dict()
report["metrics_table"] = cleaned_metrics_table

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

# Save test predictions from best model for reference
best_model = models[best_model_name]
if best_model_name == "Logistic Regression":
    preds = best_model.predict(X_test_scaled)
else:
    preds = best_model.predict(X_test)

pred_df = X_test.copy()
pred_df["ActualChurn"] = y_test.values
pred_df["PredictedChurn"] = preds
pred_df.to_csv(f"{OUTPUT_DIR}/test_predictions_{best_model_name.replace(' ', '_')}.csv", index=False)

print(json.dumps(report, indent=2, default=str))
