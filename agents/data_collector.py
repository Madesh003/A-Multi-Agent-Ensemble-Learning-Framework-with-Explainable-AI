import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# BASE PATH
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset():
    path = os.path.join(BASE_DIR, "data/district_dataset.csv")
    df = pd.read_csv(path)
    print("✅ Dataset Loaded Successfully!")
    return df


# -----------------------------
# CLEAN DATA
# -----------------------------
def clean_data(df):
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    print("Columns:", df.columns)

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def feature_engineering(df):

    # Encode location (district names → numbers)
    if 'location' in df.columns:
        le = LabelEncoder()
        df['location'] = le.fit_transform(df['location'])

    return df


# -----------------------------
# CREATE TARGET (IF NOT PRESENT)
# -----------------------------
def create_target(df):

    if 'target' not in df.columns:

        def classify(row):
            score = 0

            if row.get('rainfall', 0) > 150:
                score += 2
            if row.get('wind_speed', 0) > 80:
                score += 2
            if row.get('temperature', 0) > 40:
                score += 1
            if row.get('deaths', 0) > 20:
                score += 2

            if score >= 4:
                return 2  # High
            elif score >= 2:
                return 1  # Moderate
            else:
                return 0  # Low

        df['target'] = df.apply(classify, axis=1)
        print("✅ Target column created!")

    return df


# -----------------------------
# FINAL SAVE
# -----------------------------
def save_dataset(df):
    output_path = os.path.join(BASE_DIR, "data/final_disaster_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Final dataset saved at: {output_path}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running Data Collector...")

    df = load_dataset()

    df = clean_data(df)

    df = feature_engineering(df)

    df = create_target(df)

    print("\n📊 Preview of Processed Data:")
    print(df.head())

    save_dataset(df)

    print("✅ Data is READY for ML model!")