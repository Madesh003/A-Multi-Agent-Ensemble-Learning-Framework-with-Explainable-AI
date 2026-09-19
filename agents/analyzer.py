import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# LOAD DATA
# -----------------------------
path = os.path.join(BASE_DIR, "data/disaster_dataset.csv")
df = pd.read_csv(path)

df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df['rainfall'] = df['annual']
df['wind_speed'] = df['jun-sep'] * 0.4
df['temperature'] = df['mar-may'] * 0.3
df['deaths'] = df['rainfall'] * 0.02

df['target'] = df['target'].astype(int)

features = ['rainfall', 'wind_speed', 'temperature', 'deaths']
X = df[features]
y = df['target']

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# RANDOM FOREST (GRID SEARCH)
# -----------------------------
rf = RandomForestClassifier(n_estimators=100)

param_grid = {'max_depth': [5, 10, 15, None]}

grid = GridSearchCV(rf, param_grid, cv=3)
grid.fit(X_train, y_train)

rf_model = grid.best_estimator_
print("✅ Best RF Params:", grid.best_params_)

# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

ensemble_pred = ((rf_pred + lr_pred) / 2).round().astype(int)

# -----------------------------
# METRICS FUNCTION
# -----------------------------
def evaluate(model, X_test, y_true, y_pred, name):

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    try:
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
    except:
        auc = 0

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc,
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }

# -----------------------------
# EVALUATE MODELS
# -----------------------------
rf_metrics = evaluate(rf_model, X_test, y_test, rf_pred, "Random Forest")
lr_metrics = evaluate(lr_model, X_test, y_test, lr_pred, "Logistic Regression")

ens_metrics = {
    "Model": "Ensemble",
    "Accuracy": accuracy_score(y_test, ensemble_pred),
    "Precision": precision_score(y_test, ensemble_pred, average='weighted'),
    "Recall": recall_score(y_test, ensemble_pred, average='weighted'),
    "F1": f1_score(y_test, ensemble_pred, average='weighted'),
    "AUC": 0,
    "Confusion Matrix": confusion_matrix(y_test, ensemble_pred)
}

metrics_df = pd.DataFrame([rf_metrics, lr_metrics, ens_metrics])

print("\n📊 METRICS TABLE:\n")
print(metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']])

# Save metrics
metrics_df.to_csv(os.path.join(BASE_DIR, "data/model_metrics.csv"), index=False)

# -----------------------------
# SHAP EXPLAINABILITY
# -----------------------------
print("\n🔍 Running SHAP...")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Save SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(os.path.join(BASE_DIR, "data/shap_summary.png"))
plt.close()

print("✅ SHAP plot saved")

# -----------------------------
# UNIVERSAL FEATURE IMPORTANCE FIX
# -----------------------------
try:
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            shap_array = np.mean(np.abs(shap_array), axis=0)

    else:
        shap_array = np.abs(shap_values)

    if shap_array.ndim > 2:
        shap_array = shap_array.mean(axis=1)

    feature_importance = shap_array.mean(axis=0)

    # Ensure 1D
    feature_importance = np.ravel(feature_importance)

    # Align safely
    size = min(len(features), len(feature_importance))

    feature_df = pd.DataFrame({
        "Feature": features[:size],
        "Importance": feature_importance[:size]
    }).sort_values(by="Importance", ascending=False)

    print("\n🔥 Top Features:\n")
    print(feature_df.head(5))

except Exception as e:
    print("⚠️ SHAP feature importance skipped due to shape issue:", e)

# -----------------------------
# SAVE COMPARISON TABLE
# -----------------------------
metrics_df.to_csv(os.path.join(BASE_DIR, "data/model_comparison.csv"), index=False)

print("\n🚀 Analyzer Completed Successfully!")