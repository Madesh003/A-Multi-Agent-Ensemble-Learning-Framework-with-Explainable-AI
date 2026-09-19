import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# LOAD DATA
# -----------------------------
path = os.path.join(BASE_DIR, "data/disaster_dataset.csv")
df = pd.read_csv(path)

print("✅ Dataset loaded")

df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df['rainfall'] = df['annual']
df['wind_speed'] = df['jun-sep'] * 0.4
df['temperature'] = df['mar-may'] * 0.3
df['deaths'] = df['rainfall'] * 0.02

# -----------------------------
# TARGET FIX
# -----------------------------
df['target'] = df['target'].astype(int)

# -----------------------------
# USE ONLY FIXED FEATURES
# -----------------------------
features = ['rainfall', 'wind_speed', 'temperature', 'deaths']

X = df[features]
y = df['target']

print("Training features:", X.columns)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
model_path = os.path.join(BASE_DIR, "models/model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)

joblib.dump(model, model_path)

print("\n✅ Model saved successfully!")