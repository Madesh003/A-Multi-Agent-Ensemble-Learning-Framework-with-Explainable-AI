"""
main_agent.py — Master orchestrator for the Disaster Management Multi-Agent Pipeline

Execution Order:
  1. Data Collector  → loads, cleans, engineers features, creates target
  2. Train Model     → trains RandomForest on disaster_dataset.csv, saves model.pkl
  3. Analyzer        → evaluates models (RF + LR + Ensemble), SHAP explainability
  4. Planner         → predicts severity, allocates rescue resources
  5. Communicator    → generates alerts and full reports per district
"""

import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(BASE_DIR, "agents")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, AGENTS_DIR)
sys.path.insert(0, MODELS_DIR)

import pandas as pd
import joblib

def run_data_collector():
    print("\n" + "="*55)
    print("  STEP 1: DATA COLLECTOR AGENT")
    print("="*55)

    from agents.data_collector import (
        load_dataset, clean_data, feature_engineering,
        create_target, save_dataset
    )

    df = load_dataset()
    df = clean_data(df)
    df = feature_engineering(df)
    df = create_target(df)

    print("\n📊 Preview of Processed Data:")
    print(df.head())

    save_dataset(df)
    print("✅ Data Collector completed.\n")
    return df



def run_train_model():
    print("\n" + "="*55)
    print("  STEP 2: TRAIN MODEL")
    print("="*55)

    # Import training logic inline (train_model.py uses top-level code).
    # We replicate it here as a callable function to avoid re-running on import.
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    data_path = os.path.join(DATA_DIR, "disaster_dataset.csv")
    df = pd.read_csv(data_path)
    print("✅ Dataset loaded")

    df.columns = df.columns.str.lower().str.strip()

    df['rainfall']    = df['annual']
    df['wind_speed']  = df['jun-sep'] * 0.4
    df['temperature'] = df['mar-may'] * 0.3
    df['deaths']      = df['rainfall'] * 0.02
    df['target']      = df['target'].astype(int)

    features = ['rainfall', 'wind_speed', 'temperature', 'deaths']
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved → {model_path}\n")
    return model



def run_analyzer():
    print("\n" + "="*55)
    print("  STEP 3: ANALYZER AGENT")
    print("="*55)

    import numpy as np
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

    path = os.path.join(DATA_DIR, "disaster_dataset.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    df['rainfall']    = df['annual']
    df['wind_speed']  = df['jun-sep'] * 0.4
    df['temperature'] = df['mar-may'] * 0.3
    df['deaths']      = df['rainfall'] * 0.02
    df['target']      = df['target'].astype(int)

    features = ['rainfall', 'wind_speed', 'temperature', 'deaths']
    X = df[features]
    y = df['target']

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest with GridSearch
    rf = RandomForestClassifier(n_estimators=100)
    grid = GridSearchCV(rf, {'max_depth': [5, 10, 15, None]}, cv=3)
    grid.fit(X_train, y_train)
    rf_model = grid.best_estimator_
    print("✅ Best RF Params:", grid.best_params_)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=500)
    lr_model.fit(X_train, y_train)

    rf_pred       = rf_model.predict(X_test)
    lr_pred       = lr_model.predict(X_test)
    ensemble_pred = ((rf_pred + lr_pred) / 2).round().astype(int)

    def evaluate(model, X_test, y_true, y_pred, name):
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec  = recall_score(y_true, y_pred, average='weighted')
        f1   = f1_score(y_true, y_pred, average='weighted')
        try:
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception:
            auc = 0
        return {
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "AUC": auc,
            "Confusion Matrix": confusion_matrix(y_true, y_pred)
        }

    rf_metrics  = evaluate(rf_model,  X_test, y_test, rf_pred,       "Random Forest")
    lr_metrics  = evaluate(lr_model,  X_test, y_test, lr_pred,       "Logistic Regression")
    ens_metrics = {
        "Model": "Ensemble",
        "Accuracy":  accuracy_score(y_test,  ensemble_pred),
        "Precision": precision_score(y_test, ensemble_pred, average='weighted'),
        "Recall":    recall_score(y_test,    ensemble_pred, average='weighted'),
        "F1":        f1_score(y_test,        ensemble_pred, average='weighted'),
        "AUC": 0,
        "Confusion Matrix": confusion_matrix(y_test, ensemble_pred)
    }

    metrics_df = pd.DataFrame([rf_metrics, lr_metrics, ens_metrics])
    print("\n📊 METRICS TABLE:\n")
    print(metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']])

    metrics_df.to_csv(os.path.join(DATA_DIR, "model_metrics.csv"),     index=False)
    metrics_df.to_csv(os.path.join(DATA_DIR, "model_comparison.csv"),  index=False)

    # SHAP
    print("\n🔍 Running SHAP...")
    explainer  = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(DATA_DIR, "shap_summary.png"))
    plt.close()
    print("✅ SHAP plot saved")

    try:
        shap_array = np.array(shap_values)
        if shap_array.ndim == 3:
            shap_array = np.mean(np.abs(shap_array), axis=0)
        else:
            shap_array = np.abs(shap_array)
        if shap_array.ndim > 2:
            shap_array = shap_array.mean(axis=1)
        feature_importance = np.ravel(shap_array.mean(axis=0))
        size = min(len(features), len(feature_importance))
        feature_df = pd.DataFrame({
            "Feature": features[:size],
            "Importance": feature_importance[:size]
        }).sort_values(by="Importance", ascending=False)
        print("\n🔥 Top Features:\n")
        print(feature_df.head(5))
    except Exception as e:
        print("⚠️ SHAP feature importance skipped:", e)

    print("✅ Analyzer completed.\n")



def run_planner():
    print("\n" + "="*55)
    print("  STEP 4: PLANNER AGENT")
    print("="*55)

    from agents.planner import load_model, generate_plans

    data_path = os.path.join(DATA_DIR, "disaster_dataset.csv")
    df    = pd.read_csv(data_path)
    model = load_model()

    final_df = generate_plans(df, model)

    print("\n📊 Sample Output:")
    print(final_df[['predicted_target', 'level', 'action', 'rescue_teams']].head())

    output_path = os.path.join(DATA_DIR, "final_rescue_plan.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Rescue plan saved → {output_path}\n")



def run_communicator():
    print("\n" + "="*55)
    print("  STEP 5: COMMUNICATOR AGENT")
    print("="*55)

    from agents.communicator import (
        load_model, prepare_features,
        allocate_resources, generate_alert, generate_report
    )

    data_path = os.path.join(DATA_DIR, "disaster_dataset.csv")
    df = pd.read_csv(data_path)

    if 'affected_population' not in df.columns:
        print("⚠️ 'affected_population' missing → using default 10000")
        df['affected_population'] = 10000

    model = load_model()
    X = prepare_features(df)
    df['predicted_target'] = model.predict(X)

    results = []
    for _, row in df.iterrows():
        plan     = allocate_resources(int(row['predicted_target']),
                                      int(row['affected_population']))
        combined = {**row.to_dict(), **plan}
        results.append(combined)

    df = pd.DataFrame(results)
    df['alert']  = df.apply(generate_alert,  axis=1)
    df['report'] = df.apply(generate_report, axis=1)

    output_path = os.path.join(DATA_DIR, "final_communication_output.csv")
    df.to_csv(output_path, index=False)

    print("\n📢 Sample Alert:\n")
    print(df['alert'].iloc[0])
    print("\n📄 Sample Report:\n")
    print(df['report'].iloc[0])
    print(f"\n✅ Output saved → {output_path}")
    print("✅ Communicator completed.\n")



if __name__ == "__main__":
    print("\n" + "🌟"*28)
    print("   DISASTER MANAGEMENT — MAIN AGENT PIPELINE")
    print("🌟"*28)

    try:
        run_data_collector()   # Step 1
        run_train_model()      # Step 2
        run_analyzer()         # Step 3
        run_planner()          # Step 4
        run_communicator()     # Step 5

        print("\n" + "✅"*28)
        print("   ALL AGENTS COMPLETED SUCCESSFULLY!")
        print("✅"*28 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise