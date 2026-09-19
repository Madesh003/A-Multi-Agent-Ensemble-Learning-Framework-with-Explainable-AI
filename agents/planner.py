import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    model_path = os.path.join(BASE_DIR, "models/model.pkl")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    return model


# -----------------------------
# PREPARE FEATURES (MATCH TRAINING)
# -----------------------------
def prepare_features(df):

    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    df['rainfall'] = df['annual']
    df['wind_speed'] = df['jun-sep'] * 0.4
    df['temperature'] = df['mar-may'] * 0.3
    df['deaths'] = df['rainfall'] * 0.02

    X = df[['rainfall', 'wind_speed', 'temperature', 'deaths']]

    return X


# -----------------------------
# RESOURCE ALLOCATION
# -----------------------------
def allocate_resources(severity, population):

    if severity == 0:
        return {
            "level": "LOW",
            "action": "Monitoring",
            "rescue_teams": max(1, population // 50000),
            "ambulances": max(1, population // 100000)
        }

    elif severity == 1:
        return {
            "level": "MODERATE",
            "action": "Prepared Evacuation",
            "rescue_teams": max(5, population // 20000),
            "ambulances": max(3, population // 50000)
        }

    else:
        return {
            "level": "HIGH",
            "action": "Immediate Rescue",
            "rescue_teams": max(10, population // 10000),
            "ambulances": max(5, population // 20000)
        }


# -----------------------------
# GENERATE PLANS
# -----------------------------
def generate_plans(df, model):

    df = df.copy()

    if 'affected_population' not in df.columns:
        df['affected_population'] = 10000

    X = prepare_features(df)

    predictions = model.predict(X)
    df['predicted_target'] = predictions

    plans = []

    for _, row in df.iterrows():
        plan = allocate_resources(
            int(row['predicted_target']),
            int(row['affected_population'])
        )
        plans.append(plan)

    plan_df = pd.DataFrame(plans)

    return pd.concat([df, plan_df], axis=1)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("🚀 Running FINAL Planner...")

    data_path = os.path.join(BASE_DIR, "data/disaster_dataset.csv")
    df = pd.read_csv(data_path)

    model = load_model()

    final_df = generate_plans(df, model)

    print("\n📊 Sample Output:")
    print(final_df[['predicted_target', 'level', 'action', 'rescue_teams']].head())

    output_path = os.path.join(BASE_DIR, "data/final_rescue_plan.csv")
    final_df.to_csv(output_path, index=False)

    print("\n✅ Planner completed successfully!")