
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "traffic_congestion_advanced_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "processed_data"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading traffic dataset...")
df = pd.read_csv(DATASET_PATH)

# Exclude identifiers and target/derived labels
exclude_columns = [
    "Record_ID",
    "Timestamp",
    "Congestion_Level",
    "Anomaly_Label",
    "Congestion_Score"
]

features = df.drop(columns=exclude_columns, errors="ignore")

# Identify numeric and categorical features
numeric_columns = features.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = features.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Preprocess numeric and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Build the Isolation Forest model
anomaly_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("detector", IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42
    ))
])

print("Training Isolation Forest...")
anomaly_pipeline.fit(features)

# Predict anomalies: -1 = anomaly, 1 = normal
predictions = anomaly_pipeline.predict(features)

df["Detected_Anomaly"] = [
    "Anomaly" if prediction == -1 else "Normal"
    for prediction in predictions
]

# Save the trained model and detection results
model_path = MODEL_DIR / "isolation_forest_pipeline.joblib"
output_path = OUTPUT_DIR / "anomaly_detection_results.csv"

joblib.dump(anomaly_pipeline, model_path)
df.to_csv(output_path, index=False)

print("\nANOMALY DETECTION COMPLETED")
print(f"Total records: {len(df)}")
print(f"Detected anomalies: {(df['Detected_Anomaly'] == 'Anomaly').sum()}")
print(f"Detected normal records: {(df['Detected_Anomaly'] == 'Normal').sum()}")
print(f"Saved model to: {model_path}")
print(f"Saved results to: {output_path}")

# Compare detections with the existing labels
if "Anomaly_Label" in df.columns:
    print("\nCOMPARISON WITH EXISTING ANOMALY LABELS")

    comparison = pd.crosstab(
        df["Anomaly_Label"].astype(str).str.strip(),
        df["Detected_Anomaly"],
        rownames=["Existing Label"],
        colnames=["Isolation Forest"]
    )
    print(comparison)

    actual = (
        df["Anomaly_Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    predicted = df["Detected_Anomaly"].map({
        "Anomaly": 1,
        "Normal": 0
    })

    # Report coverage of valid existing labels
    valid = actual.notna() & predicted.notna()

    if valid.any():
        print("\nTRAINING-DATA COMPARISON REPORT")
        print("Note: This is not an independent test-set evaluation.")

        print(classification_report(
            actual[valid],
            predicted[valid],
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0
        ))

        print("Confusion matrix:")
        print(confusion_matrix(
            actual[valid],
            predicted[valid],
            labels=[0, 1]
        ))

# Show a sample of detected anomalies
print("\nSAMPLE OF DETECTED ANOMALIES")

anomaly_samples = df[df["Detected_Anomaly"] == "Anomaly"]

columns_to_show = [
    "Road_ID",
    "Junction_ID",
    "Road_Type",
    "Time_Slot",
    "Vehicle_Count",
    "Average_Speed_kmph",
    "Traffic_Density_vehicles_km",
    "Road_Occupancy_pct",
    "Average_Waiting_Time_sec",
    "Travel_Time_Ratio",
    "Anomaly_Label"
]

available_columns = [
    column for column in columns_to_show if column in df.columns
]

print(anomaly_samples[available_columns].head(10).to_string(index=False))

print("\nAnomaly detection finished successfully.")
