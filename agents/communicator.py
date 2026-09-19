import pandas as pd
import joblib
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    model_path = os.path.join(BASE_DIR, "models/model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ model.pkl not found. Run training first.")

    print("✅ Model loaded")
    return joblib.load(model_path)


# -----------------------------
# PREPARE FEATURES (MATCH TRAINING EXACTLY)
# -----------------------------
def prepare_features(df):

    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Safety checks
    required_cols = ['annual', 'jun-sep', 'mar-may']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    df['rainfall'] = df['annual']
    df['wind_speed'] = df['jun-sep'] * 0.4
    df['temperature'] = df['mar-may'] * 0.3
    df['deaths'] = df['rainfall'] * 0.02

    return df[['rainfall', 'wind_speed', 'temperature', 'deaths']]


# -----------------------------
# RESOURCE ALLOCATION
# -----------------------------
def allocate_resources(severity, population):

    if population <= 0:
        population = 1

    severity_map = {
        0: ("LOW", "Monitoring", 0.5),
        1: ("MODERATE", "Prepared Evacuation", 1.0),
        2: ("HIGH", "Immediate Rescue", 2.0)
    }

    level, action, weight = severity_map.get(severity, ("LOW", "Monitoring", 0.5))

    rescue_teams = max(1, int((population / 10000) * weight))
    ambulances = max(1, int((population / 20000) * weight))
    food_kits = int(population * (0.1 * weight))
    camps = max(1, int((population / 25000) * weight))
    medical_teams = max(1, int((population / 30000) * weight))
    water_units = max(1, int((population / 20000) * weight))

    return {
        "level": level,
        "action": action,
        "rescue_teams": rescue_teams,
        "ambulances": ambulances,
        "food_kits": food_kits,
        "camps": camps,
        "medical_teams": medical_teams,
        "water_units": water_units
    }


# -----------------------------
# ALERT GENERATOR
# -----------------------------
def generate_alert(row):

    return f"""
🚨 DISASTER ALERT 🚨
--------------------------------
Location            : {row.get('subdivision', 'Unknown')}
Severity Level      : {row['level']}

⚠️ Action Required  :
{row['action']}

Stay Safe!
--------------------------------
""".strip()


# -----------------------------
# REPORT GENERATOR
# -----------------------------
def generate_report(row):

    return f"""
================ DISASTER REPORT ================

📍 Location              : {row.get('subdivision', 'Unknown')}
👥 Affected Population  : {row.get('affected_population', 0)}

🔥 Severity Level        : {row['level']}
📊 Prediction Class     : {row['predicted_target']}

🛠️ Action Plan          : {row['action']}

🚑 Resources Allocated:
   - Rescue Teams       : {row['rescue_teams']}
   - Ambulances         : {row['ambulances']}
   - Food Kits          : {row['food_kits']}
   - Relief Camps       : {row['camps']}
   - Medical Teams      : {row['medical_teams']}
   - Water Units        : {row['water_units']}

🕒 Generated At         : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

=================================================
""".strip()


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    print("🚀 Running FINAL Communicator...")

    data_path = os.path.join(BASE_DIR, "data/disaster_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError("❌ disaster_dataset.csv not found")

    df = pd.read_csv(data_path)

    # -----------------------------
    # FIX: AUTO CREATE POPULATION IF MISSING
    # -----------------------------
    if 'affected_population' not in df.columns:
        print("⚠️ 'affected_population' missing → creating default values")
        df['affected_population'] = 10000

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    model = load_model()

    # -----------------------------
    # PREPARE FEATURES
    # -----------------------------
    X = prepare_features(df)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    df['predicted_target'] = model.predict(X)

    # -----------------------------
    # RESOURCE PLANNING
    # -----------------------------
    results = []

    for _, row in df.iterrows():

        plan = allocate_resources(
            int(row['predicted_target']),
            int(row['affected_population'])
        )

        combined = {**row.to_dict(), **plan}
        results.append(combined)

    df = pd.DataFrame(results)

    # -----------------------------
    # GENERATE OUTPUT
    # -----------------------------
    df['alert'] = df.apply(generate_alert, axis=1)
    df['report'] = df.apply(generate_report, axis=1)

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    output_path = os.path.join(BASE_DIR, "data/final_communication_output.csv")
    df.to_csv(output_path, index=False)

    # -----------------------------
    # DISPLAY SAMPLE
    # -----------------------------
    print("\n📢 Sample Alert:\n")
    print(df['alert'].iloc[0])

    print("\n📄 Sample Report:\n")
    print(df['report'].iloc[0])

    print(f"\n✅ Output saved at: {output_path}")
    print("✅ Communicator completed successfully!")