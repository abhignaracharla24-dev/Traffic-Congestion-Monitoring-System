
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "traffic_congestion_advanced_dataset.csv"
)

FEATURES = [
    "Vehicle_Count",
    "Average_Speed_kmph",
    "Traffic_Density_vehicles_km",
    "Road_Occupancy_pct",
    "Average_Waiting_Time_sec",
    "Travel_Time_Ratio",
]

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

y = (
    df["Anomaly_Label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("yes")
    .astype(int)
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Known anomalies in test set:", int(y_test.sum()))

for contamination in [0.03, 0.05, 0.08, 0.10]:
    detector = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )

    detector.fit(X_train)

    # Evaluate only on records not used to fit the detector.
    predictions = (detector.predict(X_test) == -1).astype(int)

    print(f"\nCONTAMINATION: {contamination}")
    print("Detected anomalies:", int(predictions.sum()))
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
