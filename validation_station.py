# ==============================================================================
# CROSS-DATASET VALIDATION: trained on city_day.csv -> tested on station_day.csv
# Run aqi_main.py FIRST, it saves the models this file loads.
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix

# 1. Load the saved models, scaler and settings from the main training run
rf_reg = joblib.load("best_regressor_rf.pkl")
final_clf = joblib.load("best_classifier.pkl")              # the final model
scaler = joblib.load("feature_Scaler.pkl")
features = joblib.load("features.pkl")                      # same columns, same order as training
train_medians = joblib.load("train_medians.pkl")            # medians from the city_day TRAINING data
city_info = joblib.load("city_day_info.pkl")                # for the dataset comparison table

# 2. Load station-level dataset
df_station = pd.read_csv("station_day.csv")
print(f"Station Day Initial Shape: {df_station.shape}")

# 3. Preprocessing & Temporal Lag Engineering (grouped by StationId) - same steps as the training file
df = df_station.copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["StationId", "Date"])

df["Previous AQI"] = df.groupby("StationId")["AQI"].shift(1)
df["Next AQI"] = df.groupby("StationId")["AQI"].shift(-1)
df["Previous Date"] = df.groupby("StationId")["Date"].shift(1)
df["Next Date"] = df.groupby("StationId")["Date"].shift(-1)

consecutive_condition = (
    (df["Date"] - df["Previous Date"] == pd.Timedelta(days=1)) &
    (df["Next Date"] - df["Date"] == pd.Timedelta(days=1))
)

df = df[consecutive_condition].copy()
df["Month"] = df["Date"].dt.month                           # calendar features, like training
df["DayOfWeek"] = df["Date"].dt.dayofweek

features_5 = ["AQI", "Previous AQI", "PM2.5", "NO2", "CO"]
df = df.dropna(subset=features_5 + ["Next AQI"]).copy()
df = df[(df["AQI"] <= 500) & (df["Next AQI"] <= 500)].copy()    # remove impossible AQI values, like training
df["Unhealthy Next Day"] = (df["Next AQI"] > 100).astype(int)

# Fill missing values of the extra columns with the TRAINING medians (never with station_day's own)
for column, median in train_medians.items():
    df[column] = df[column].fillna(median)

X_ext = df[features].copy()
y_reg_test = df["Next AQI"]
y_clf_test = df["Unhealthy Next Day"]

# 4. Scale with the SAVED scaler from training (a new scaler would not match what the models learned)
X_ext_scaled = scaler.transform(X_ext)

# 5. Regression: predict next-day AQI
reg_preds = rf_reg.predict(X_ext_scaled)
baseline_rmse = np.sqrt(mean_squared_error(y_reg_test, df["AQI"]))      # baseline: tomorrow = today
rf_rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
print(f"\nRegression RMSE  Baseline (tomorrow = today): {baseline_rmse:.2f}   Random Forest: {rf_rmse:.2f}")

# 6. Classification: unhealthy tomorrow?
clf_preds = final_clf.predict(X_ext_scaled)
rule_preds = (df["AQI"] > 100).astype(int)                               # baseline rule: today > 100

results = []
for name, preds in [("Baseline rule (today>100)", rule_preds), ("Final model", clf_preds)]:
    results.append([name,
                    round(accuracy_score(y_clf_test, preds),4),
                    round(precision_score(y_clf_test, preds),4),
                    round(recall_score(y_clf_test, preds),4),
                    round(f1_score(y_clf_test, preds),4)])

print("\nClassification on station_day.csv:")
print(pd.DataFrame(results, columns=["Method","Accuracy","Precision","Recall","F1"]).to_string(index=False))
print(f"\nFinal model confusion matrix:\n{confusion_matrix(y_clf_test, clf_preds)}")

# 7. Dataset comparison table
dataset_table = pd.DataFrame([
    ["city_day.csv (primary)", city_info["rows"], len(features), "Unhealthy next day", city_info["unhealthy_pct"]],
    ["station_day.csv (external)", len(df), len(features), "Unhealthy next day", round(y_clf_test.mean()*100,2)]
], columns=["Dataset","Usable Rows","Features Used","Target","Unhealthy %"])
print("\n", dataset_table.to_string(index=False))

print("\nCross-Dataset Validation using Saved Models Completed Successfully.")
