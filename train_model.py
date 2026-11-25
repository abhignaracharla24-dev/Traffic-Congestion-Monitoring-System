import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "traffic_congestion_advanced_dataset.csv"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("TRAFFIC CONGESTION MONITORING SYSTEM")
print("MACHINE LEARNING MODEL TRAINING")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH, keep_default_na=False)

TARGET = "Congestion_Level"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

# Remove rows with missing target labels
df = df[df[TARGET].astype(str).str.strip() != ""].copy()

# Separate features and target
X = df.drop(
    columns=[TARGET, "Anomaly_Label", "Congestion_Score"],
    errors="ignore"
)
y = df[TARGET].astype(str).str.strip()

# Remove identifier columns that should not be model features
id_columns = [
    col for col in X.columns
    if col.lower() in ["id", "record_id", "submission_id"]
]
X = X.drop(columns=id_columns, errors="ignore")

# Identify numeric and categorical features
numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
categorical_columns = [
    col for col in X.columns if col not in numeric_columns
]

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build and train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

print(f"\nTotal records: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Input features: {X.shape[1]}")

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
weighted_f1 = f1_score(
    y_test, predictions, average="weighted", zero_division=0
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save trained model
model_path = OUTPUT_DIR / "congestion_model.joblib"
joblib.dump(model, model_path)

# Save feature names for the dashboard
metadata = {
    "feature_columns": X.columns.tolist(),
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "target_column": TARGET,
    "class_labels": sorted(y.unique().tolist())
}
joblib.dump(metadata, OUTPUT_DIR / "model_metadata.joblib")

print(f"\nModel saved to: {model_path}")
print(f"Metadata saved to: {OUTPUT_DIR / 'model_metadata.joblib'}")
print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
