
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "dataset" / "traffic_congestion_advanced_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, keep_default_na=False)

print("=" * 55)
print("TRAFFIC DATA PREPROCESSING")
print("=" * 55)

# Convert timestamp to datetime
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], errors="coerce"
)

# Convert blank strings to missing values
df = df.replace(r"^\s*$", np.nan, regex=True)

# Remove duplicate record IDs
df = df.drop_duplicates(subset="Record_ID")

# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

numeric_features = [
    "Vehicle_Count",
    "Average_Speed_kmph",
    "Traffic_Density_vehicles_km",
    "Road_Occupancy_pct",
    "Average_Waiting_Time_sec",
    "Travel_Time_Ratio",
]

categorical_features = [
    "Road_ID",
    "Junction_ID",
    "Road_Type",
    "Day_of_Week",
    "Time_Slot",
    "Weather",
    "Incident_Type",
]

target = "Congestion_Level"

# Convert numerical features safely
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows without essential target/time information
df = df.dropna(
    subset=["Timestamp", target, "Record_ID"]
).copy()

# Ensure all four expected labels are valid
valid_labels = ["Normal", "Moderate", "High", "Severe"]
df = df[df[target].isin(valid_labels)].copy()

# Sort chronologically
df = df.sort_values(
    ["Timestamp", "Record_ID"]
).reset_index(drop=True)

# --------------------------------------------------
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df[numeric_features + categorical_features]
X_test = test_df[numeric_features + categorical_features]

y_train = train_df[target]
y_test = test_df[target]

# --------------------------------------------------
# 5. PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# Fit only on training data to avoid data leakage
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --------------------------------------------------
# 6. SAVE PROCESSED DATA
# --------------------------------------------------

joblib.dump(
    preprocessor,
    OUTPUT_DIR / "preprocessor.joblib"
)

joblib.dump(
    numeric_features,
    OUTPUT_DIR / "numeric_features.joblib"
)

joblib.dump(
    categorical_features,
    OUTPUT_DIR / "categorical_features.joblib"
)

# Save matrices in a format suitable for later model training
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

np.save(OUTPUT_DIR / "X_train.npy", X_train_processed)
np.save(OUTPUT_DIR / "X_test.npy", X_test_processed)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

# Keep raw chronological subsets for time-series experiments
train_df.to_csv(OUTPUT_DIR / "train_raw.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_raw.csv", index=False)

# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\nOriginal dataset records:", len(df))
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining matrix shape:", X_train_processed.shape)
print("Testing matrix shape:", X_test_processed.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

print("\nSaved preprocessing pipeline and datasets to:")
print(OUTPUT_DIR)

print("\nSTEP 7 COMPLETED SUCCESSFULLY!")
