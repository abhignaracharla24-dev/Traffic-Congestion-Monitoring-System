TRAFFIC CONGESTION MONITORING SYSTEM
=====================================

PROJECT OVERVIEW
----------------
The Traffic Congestion Monitoring System is a machine-learning-based traffic
analysis and monitoring application developed to identify congestion levels
from historical traffic data.

The system analyzes traffic-related parameters such as vehicle count,
average speed, traffic density, road occupancy, average waiting time,
travel time ratio, road information, weather, incidents, and time-related
attributes. Based on these inputs, the system classifies traffic conditions
into four congestion levels:

    Normal
    Moderate
    High
    Severe

The project also includes an anomaly detection component for identifying
unusual traffic patterns and a Streamlit-based dashboard for viewing traffic
analytics, predictions, anomaly information, dataset records, and model
performance.

This project is implemented as a dataset-based prototype. The current
dashboard does not use a live traffic sensor or real-time traffic feed.

PROJECT STRUCTURE
----------------
The main project folder contains the following important files and folders:

    dataset/
        Contains the traffic dataset used by the project.

    processed_data/
        Stores processed datasets and generated results produced during
        preprocessing and anomaly detection.

    models/
        Stores the trained machine-learning model, model metadata,
        preprocessing pipeline, and anomaly-detection model.

    preprocess_data.py
        Performs data cleaning, feature preparation, chronological
        train-test splitting, preprocessing, and saves the processed data.

    train_model.py
        Builds and trains the Random Forest congestion classification model,
        evaluates its performance, and saves the trained model and metadata.

    anomaly_detection.py
        Builds an Isolation Forest anomaly-detection pipeline, detects
        unusual traffic records, compares the detected results with the
        existing anomaly labels, and saves the results.

    evaluate_anomaly_detection.py
        Evaluates anomaly detection separately using a train-test split and
        tests different contamination values for the Isolation Forest model.

    app.py
        Runs the Streamlit dashboard through which the project's results and
        functionality can be viewed.

    requirements.txt
        Contains the Python libraries required to run the project.

    README.txt
        Provides instructions and an overview of the project.

DATASET
-------
The project uses:

    dataset/traffic_congestion_advanced_dataset.csv

The dataset contains 1,200 traffic records and 19 columns. The main traffic
attributes include vehicle count, average speed, traffic density, road
occupancy, average waiting time, travel time ratio, congestion score,
congestion level, and anomaly label.

The target variable for congestion classification is:

    Congestion_Level

The possible congestion levels are:

    Normal, Moderate, High, Severe

The project also contains:

    Anomaly_Label

which indicates whether a record has been labelled as an anomaly.

IMPORTANT NOTE ABOUT CONGESTION SCORE
-------------------------------------
Congestion_Score is available in the original dataset, but it is excluded
from the Random Forest classification model to avoid target-related leakage.
The model instead learns the congestion level from the other available
traffic and contextual features.

INSTALLATION
------------
1. Open the project folder in VS Code.

2. Open a terminal in the project folder.

3. Create a virtual environment if one has not already been created:

       python -m venv venv

4. Activate the virtual environment on Windows PowerShell:

       .\venv\Scripts\Activate.ps1

   If PowerShell blocks the activation script, the project can also be run
   from a Command Prompt after activating the environment with:

       venv\Scripts\activate

5. Install the required libraries:

       pip install -r requirements.txt

RUNNING THE PROJECT
-------------------
The project is designed to be executed in stages. Running the files in the
following order makes it easier to generate the required processed data,
models, and results.

STEP 1: PREPROCESS THE DATA
---------------------------
Run:

       python preprocess_data.py

This step prepares the traffic dataset for machine learning. It performs
basic data cleaning, handles missing values, converts the timestamp into
the required form, removes duplicate record IDs, validates congestion
labels, sorts the records chronologically, and creates the training and
testing portions of the data.

The preprocessing pipeline also prepares numerical and categorical features
using imputation, scaling for numerical features, and one-hot encoding for
categorical features.

The generated preprocessing files are stored inside:

       processed_data/

STEP 2: TRAIN THE CONGESTION CLASSIFICATION MODEL
-------------------------------------------------
Run:

       python train_model.py

This script prepares the features required for congestion classification
and trains a Random Forest Classifier.

The model uses the following traffic-related numerical information:

       Vehicle_Count
       Average_Speed_kmph
       Traffic_Density_vehicles_km
       Road_Occupancy_pct
       Average_Waiting_Time_sec
       Travel_Time_Ratio

It also uses the relevant categorical road, junction, time, weather, and
incident information.

The trained model and its metadata are saved in:

       models/

The main model file is:

       models/congestion_model.joblib

The metadata file is:

       models/model_metadata.joblib

STEP 3: RUN ANOMALY DETECTION
-----------------------------
Run:

       python anomaly_detection.py

This script uses an Isolation Forest model to identify traffic records that
show unusual patterns.

The anomaly-detection pipeline uses the available traffic and contextual
features while excluding the congestion target, existing anomaly label,
record identifier, timestamp, and congestion score from the model input.

The trained anomaly-detection pipeline is saved as:

       models/isolation_forest_pipeline.joblib

The generated anomaly results are saved as:

       processed_data/anomaly_detection_results.csv

The script also compares the detected anomalies with the existing
Anomaly_Label values in the dataset. This comparison is provided as an
additional analysis and should not be interpreted as an independent
test-set evaluation.

STEP 4: EVALUATE ANOMALY DETECTION
----------------------------------
Run:

       python evaluate_anomaly_detection.py

This script performs a separate evaluation of Isolation Forest anomaly
detection. It divides the available labelled data into training and testing
sets and evaluates different contamination settings.

The contamination values tested by the script are:

       0.03
       0.05
       0.08
       0.10

The evaluation produces classification metrics and a confusion matrix for
the anomaly-detection experiment.

STEP 5: START THE STREAMLIT DASHBOARD
-------------------------------------
Run:

       streamlit run app.py

After Streamlit starts, open the local address shown in the terminal,
normally:

       http://localhost:8501

The dashboard provides several sections for exploring the project.

DASHBOARD SECTIONS
------------------
Command Center
    Provides an overall view of traffic conditions, high-risk records,
    hotspots, recent traffic information, and severity-related summaries.

Traffic Analytics
    Provides charts and filtering options for analyzing congestion records
    according to traffic conditions and other available attributes.

Congestion Prediction
    Allows traffic-related input values to be passed to the trained model
    and displays the predicted congestion level and model-related scores.

Anomaly Monitoring
    Provides information about anomaly-labelled traffic records available
    in the dataset.

Dataset Explorer
    Allows the dataset to be viewed, searched, filtered, and downloaded
    through the dashboard.

Model Performance
    Displays the saved congestion-classification performance information,
    including accuracy, F1 score, confusion matrix, and class-level
    classification results.

THE ASPECTS
    Contains the project's functional requirements, non-functional
    requirements, and business/product-oriented information included in
    the prototype.

MODEL PERFORMANCE
-----------------
The congestion classification model was evaluated after excluding
Congestion_Score from the model features.

The recorded evaluation results used in the dashboard are:

    Accuracy: 91.25%
    Weighted F1 Score: 91.22%

The project also contains a confusion matrix and class-level precision,
recall, F1-score, and support values in the model-performance section of
the dashboard.

These results represent the saved model evaluation used by the current
project version.

IMPORTANT PROJECT NOTES
-----------------------
1. The project is currently a dataset-based prototype and does not require
   a live traffic sensor connection.

2. The dashboard uses the available historical traffic dataset to provide
   traffic analysis and prediction.

3. The anomaly-detection script and the separate anomaly-evaluation script
   serve different purposes. The first generates an Isolation Forest
   pipeline and anomaly results, while the second performs a separate
   evaluation experiment using labelled data.

4. The preprocessing script creates processed train/test data, while the
   current training script performs its own model-training split. Therefore,
   the preprocessing outputs should not be assumed to be the exact data
   used internally by the current training script.

5. The generated files inside the models/ and processed_data/ folders may
   change when the corresponding scripts are executed again.

6. Make sure the dataset path remains:

       dataset/traffic_congestion_advanced_dataset.csv

TROUBLESHOOTING
---------------
If Python is not recognized:
    Check that Python is installed and available in the system PATH.

If the virtual environment is not activated:
    Run the activation command again from the project root.

If a module is missing:
    Make sure the virtual environment is active and run:

       pip install -r requirements.txt

If Streamlit does not start:
    Confirm that Streamlit is installed and run:

       streamlit run app.py

If a model file is missing:
    Run the corresponding training or detection script before starting
    the dashboard.

RECOMMENDED EXECUTION ORDER
---------------------------
For a fresh setup, use this order:

    1. Activate the virtual environment
    2. Install requirements
    3. Run preprocess_data.py
    4. Run train_model.py
    5. Run anomaly_detection.py
    6. Run evaluate_anomaly_detection.py
    7. Run streamlit run app.py

PROJECT PURPOSE
---------------
The overall purpose of the project is to demonstrate how machine learning
can be applied to traffic data to classify congestion severity, identify
unusual traffic patterns, and present the resulting information through
an interactive monitoring dashboard.

The system is intended as an academic prototype that brings together data
preprocessing, supervised machine learning, anomaly detection, model
evaluation, and dashboard-based visualization in a single project.
