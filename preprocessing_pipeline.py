import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("🚀 Loading dataset...")

df = pd.read_csv("MASTER_DISASTER_DATASET.csv")

print("📊 Initial Shape:", df.shape)

# -----------------------------
# 1️⃣ HANDLE MISSING VALUES
# -----------------------------
print("🧹 Handling missing values...")

# Numeric columns
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# -----------------------------
# 2️⃣ FEATURE ENGINEERING
# -----------------------------
print("⚙️ Feature engineering...")

# Create total impact feature
df["total_impact"] = (
    df["magnitude"] +
    df["rainfall"] +
    df["wind_speed"] +
    df["water_level"]
)

# Create risk index
df["risk_index"] = (
    df["total_impact"] * (df["population_density"] + 1)
)

# -----------------------------
# 3️⃣ WEIGHTED SEVERITY SCORE 🔥
# -----------------------------
print("🔥 Calculating Weighted Severity Score...")

df["weighted_severity"] = (
    0.4 * df["magnitude"] +
    0.2 * df["rainfall"] +
    0.2 * df["wind_speed"] +
    0.1 * df["water_level"] +
    0.1 * df["population_density"]
)

# Convert into classification labels
def severity_label(x):
    if x < 1:
        return 0   # Low
    elif x < 3:
        return 1   # Medium
    else:
        return 2   # High

df["severity_label"] = df["weighted_severity"].apply(severity_label)

# -----------------------------
# 4️⃣ ENCODE CATEGORICAL DATA
# -----------------------------
print("🔤 Encoding categorical features...")

df = pd.get_dummies(df, columns=["disaster_type"], drop_first=True)

# -----------------------------
# 5️⃣ NORMALIZATION (MinMaxScaler)
# -----------------------------
print("📏 Normalizing data...")

scaler = MinMaxScaler()

features_to_scale = [
    "magnitude", "rainfall", "wind_speed",
    "water_level", "population_density",
    "total_impact", "risk_index", "weighted_severity"
]

df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# -----------------------------
# 6️⃣ FINAL CLEANUP
# -----------------------------
print("🧼 Final cleanup...")

df.replace([np.inf, -np.inf], 0, inplace=True)

# -----------------------------
# 7️⃣ SAVE DATASET
# -----------------------------
df.to_csv("preprocessed_master_dataset.csv", index=False)

print("✅ Preprocessing completed!")
print("📊 Final Shape:", df.shape)
print(df.head())