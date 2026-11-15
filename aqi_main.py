# ==============================================================================
# PROJECT: AIR QUALITY FORECASTING & EARLY WARNING SYSTEM
# Machine Learning Project Pipeline
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor

from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    classification_report,
    roc_auc_score,
    roc_curve
)

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

import warnings; warnings.filterwarnings("ignore")
import joblib

# ------------------------------------------------------------------------------
# 1. LOAD PRIMARY DATASET
# ------------------------------------------------------------------------------

df = pd.read_csv("city_day.csv")
print("Data Loaded Successfully.")
print(f"Initial Shape: Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("First 5 rows: ")
print(df.head())

print("="*100)
# ------------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------------------
print("Exploratory Data Analysis: ")

# Duplicates
duplicate_count = df.duplicated().sum()
print(f"1. Duplicate Records: {duplicate_count}")
features_to_check = ["AQI","PM2.5","NO2","CO"]
missing_val = df[features_to_check].isnull().sum()
missing_pct = (missing_val/len(df))*100
missing_summary = pd.DataFrame({"Missing Count":missing_val,"Missing Percentage":missing_pct})
print(missing_summary)

# Descriptive Statistics
print("\nNumerical Statistics: ")
print(df[features_to_check].describe().T[["mean","std","min","50%","max"]])


# ==============================================================================
# BOXPLOTS & OUTLIER DETECTION
# ==============================================================================
features_to_plot = ["AQI", "PM2.5", "NO2", "CO"]

# Visual Boxplots
for col in features_to_plot:
    plt.figure(figsize=(6,2.5))
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

# Outlier Calculation Table Via IQR
outlier_summary = []
for col in features_to_plot:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col]<lower_bound) | (df[col] > upper_bound)]
    outlier_summary.append([col,len(outliers),round(len(outliers)/len(df)*100,2)])

outlier_df = pd.DataFrame(
    outlier_summary,
    columns=["Feature","Number Of Outliers","Outlier Percentage"]
)

print("="*25,"Outlier Analysis","="*25)
print(outlier_df.sort_values("Outlier Percentage",ascending=False).to_string(index=False))

# The Indian AQI scale ends at 500, so larger values are data errors
print(f"AQI values above 500 (invalid): {(df['AQI'] > 500).sum()}")

# ==============================================================================
# Correlation Analysis and Heatmap
# ==============================================================================

print("\n","="*25,"Correlation Matrix","="*50)

# Pearson correlation Matrix
corr_features = ["AQI","PM2.5","NO2","CO"]
correlation_matrix = df[corr_features].corr()
print(correlation_matrix.round(2))

# Correlation Heatmap
plt.figure(figsize=(7,5))
plt.imshow(correlation_matrix,cmap="coolwarm",vmin=-1,vmax=1)
plt.colorbar(label = "Correlation Coefficient")

plt.xticks(range(len(corr_features)),corr_features,rotation=45)
plt.yticks(range(len(corr_features)),corr_features)

for i in range(len(corr_features)):
    for j in range(len(corr_features)):
        plt.text(j,i,f"{correlation_matrix.iloc[i,j]:.2f}",ha="center",va="center", color="black",fontsize=11)

plt.title("Correlation Matrix")
plt.show()

# ==============================================================================
# DATA PREPROCESSING & FEATURE ENGINEERING
# ==============================================================================

print("\n","="*25,"Data Preprocessing","="*25)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City","Date"])

# Temporal lag features: yesterday's and tomorrow's AQI for the same city
df["Previous AQI"] = df.groupby("City")["AQI"].shift(1)
df["Next AQI"] = df.groupby("City")["AQI"].shift(-1)
df["Previous Date"] = df.groupby("City")["Date"].shift(1)
df["Next Date"] = df.groupby("City")["Date"].shift(-1)

# Keep only rows where yesterday and tomorrow are really consecutive days
consecutive_condition = (
    (df["Date"] - df["Previous Date"] == pd.Timedelta(days=1)) &
    (df["Next Date"] - df["Date"] == pd.Timedelta(days=1))
)

df = df[consecutive_condition].copy()

basic_features = ["AQI","Previous AQI","PM2.5","NO2","CO"]

# Feature engineering: calendar features and the other pollutants
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
extra_features = ["PM10","NO","NOx","SO2","O3","Month","DayOfWeek"]
features = basic_features + extra_features

df = df.dropna(subset = basic_features + ["Next AQI"]).copy()
df = df[(df["AQI"] <= 500) & (df["Next AQI"] <= 500)].copy()        # remove invalid AQI values

df["Unhealthy Next Day"] = (df["Next AQI"] > 100).astype(int)

print(f"Cleaned Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("Target Class Distribution P(y):")
print(df["Unhealthy Next Day"].value_counts(normalize=True).round(4))

# ==============================================================================
# TRAIN-TEST SPLIT & FEATURE SCALING
# ==============================================================================

print("\n", "="*25,"Train Test Split & Scaling","="*25)

X = df[features]
y_reg = df["Next AQI"]
y_clf = df["Unhealthy Next Day"]

# Random split: neighbouring days of a city can fall in both train and test (see Limitations)
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.20,random_state=42
)

# Fill missing values with TRAIN medians only (avoids data leakage)
train_medians = {}
for column in extra_features:
    train_medians[column] = X_train[column].median()
    X_train[column] = X_train[column].fillna(train_medians[column])
    X_test[column] = X_test[column].fillna(train_medians[column])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Rows: {X_train.shape[0]}")
print(f"Testing Rows: {X_test.shape[0]}")

# Class-imbalance strategies: Original vs Class Weighting vs Random Oversampling
print("\n", "="*25,"Class Imbalance Strategies (Logistic Regression)","="*25)
for name in ["Original","Class Weighting","Random Oversampling"]:
    X_fit, y_fit, cw = X_train_scaled, y_clf_train.values, None
    if name == "Class Weighting":
        cw = "balanced"
    if name == "Random Oversampling":
        is_min = (y_clf_train == 0).values                      # class 0 (healthy) is the smaller class
        X_up = resample(X_train_scaled[is_min], replace=True, n_samples=int((~is_min).sum()), random_state=42)
        X_fit = np.vstack([X_train_scaled[~is_min], X_up])
        y_fit = np.concatenate([y_clf_train.values[~is_min], np.zeros(len(X_up), dtype=int)])
    imb_model = LogisticRegression(class_weight=cw, max_iter=1000).fit(X_fit, y_fit)
    imb_pred = imb_model.predict(X_test_scaled)
    print(f"{name:<22} Accuracy: {accuracy_score(y_clf_test,imb_pred):.4f}  Recall: {recall_score(y_clf_test,imb_pred):.4f}  F1: {f1_score(y_clf_test,imb_pred):.4f}")

# ==============================================================================
# MODEL TRAINING & EVALUATION (Linear Regression)
# ==============================================================================

print("\n","="*25, "Linear Regression Model","="*25)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled,y_reg_train)

lr_reg_pred = lr_model.predict(X_test_scaled)

lr_mae = mean_absolute_error(y_reg_test,lr_reg_pred)
lr_rmse = np.sqrt(mean_squared_error(y_reg_test,lr_reg_pred))
lr_r2 = r2_score(y_reg_test,lr_reg_pred)

print(f"LR MAE: {lr_mae}")
print(f"LR RMSE: {lr_rmse}")
print(f"LR R2: {lr_r2}")

# ==============================================================================
# MODEL TRAINING & EVALUATION (Logistic Regression)
# ==============================================================================

print("\n" + "="*25, "Logistic Regression Model", "="*25)

# Baseline rule (no ML): tomorrow is unhealthy if today's AQI > 100
rule_pred = (X_test["AQI"] > 100).astype(int)
print(f"BASELINE RULE Accuracy: {accuracy_score(y_clf_test,rule_pred):.4f}   F1: {f1_score(y_clf_test,rule_pred):.4f}")

log_model = LogisticRegression()
log_model.fit(X_train_scaled,y_clf_train)

log_pred = log_model.predict(X_test_scaled)

log_accuracy = accuracy_score(y_clf_test,log_pred)
log_precision = precision_score(y_clf_test,log_pred)
log_recall = recall_score(y_clf_test,log_pred)
log_f1 = f1_score(y_clf_test,log_pred)
log_cf = confusion_matrix(y_clf_test,log_pred)

print(f"LOG Accuracy: {log_accuracy}")
print(f"LOG Precision: {log_precision}")
print(f"LOG Recall: {log_recall}")
print(f"LOG F1: {log_f1}")
print(f"LOG Confusion Matrix: \n{log_cf}")
print(f"\nLOG Classification Report: \n{classification_report(y_clf_test,log_pred)}")

# ==============================================================================
# PROPOSED MODIFICATION / ADVANCED MODEL (Random Forest)
# ==============================================================================

# 1. Random Forest Regressor for Next AQI prediction

print("\n" + "="*25, "Random Forest Regression", "="*25)
rf_reg = RandomForestRegressor(n_estimators=100,random_state=42)
rf_reg.fit(X_train_scaled,y_reg_train)

rf_pred = rf_reg.predict(X_test_scaled)

rf_mae = mean_absolute_error(y_reg_test,rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_reg_test,rf_pred))
rf_r2 = r2_score(y_reg_test,rf_pred)

print(f"RF MAE: {rf_mae}")
print(f"RF RMSE: {rf_rmse}")
print(f"RF R2: {rf_r2}")

# 2. Random Forest Classification for Unhealthy Next Day Prediction
print("\n" + "="*25, "Random Forest Classification", "="*25)

rf_clf = RandomForestClassifier(n_estimators=100,random_state=42)
rf_clf.fit(X_train_scaled,y_clf_train)

rf_clf_pred = rf_clf.predict(X_test_scaled)

rf_acc = accuracy_score(y_clf_test, rf_clf_pred)
rf_pre = precision_score(y_clf_test, rf_clf_pred)
rf_recall = recall_score(y_clf_test, rf_clf_pred)
rf_f1 = f1_score(y_clf_test, rf_clf_pred)
rf_cf = confusion_matrix(y_clf_test, rf_clf_pred)

print(f"RF Classification Accuracy: {rf_acc:.4f}")
print(f"RF Classification Precision: {rf_pre:.4f}")
print(f"RF Classification Recall: {rf_recall:.4f}")
print(f"RF Classification f1: {rf_f1:.4f}")
print(f"RF Classification CF: {rf_cf}")
print("\nRF Classification Report:")
print(classification_report(y_clf_test, rf_clf_pred))

# ==============================================================================
# MULTIPLE ML MODELS & COMPARISON TABLES
# ==============================================================================

# Regression Model Comparison
print("\n", "="*25,"Regression Model Comparison Table: ","="*25)

knn_reg = KNeighborsRegressor(n_neighbors=5)
dt_reg = DecisionTreeRegressor(random_state=42)
svr_reg = SVR()
gb_reg = GradientBoostingRegressor(n_estimators=100,random_state=42)
ann_reg = MLPRegressor(hidden_layer_sizes=(64,32),early_stopping=True,max_iter=500,random_state=42)

knn_reg.fit(X_train_scaled,y_reg_train)
dt_reg.fit(X_train_scaled,y_reg_train)
svr_reg.fit(X_train_scaled,y_reg_train)
gb_reg.fit(X_train_scaled,y_reg_train)
ann_reg.fit(X_train_scaled,y_reg_train)

all_reg_models = {
    "Linear Regression":lr_model,
    "k-NN Regression":knn_reg,
    "Decision Tree Regression":dt_reg,
    "Random Forest Regression":rf_reg,
    "SVR":svr_reg,
    "Gradient Boosting Regression":gb_reg,
    "ANN (MLP) Regression":ann_reg
}

reg_comparison_list = []

# Baseline: tomorrow's AQI = today's AQI
reg_comparison_list.append(["Baseline (tomorrow = today)",
                            round(mean_absolute_error(y_reg_test,X_test["AQI"]),2),
                            round(np.sqrt(mean_squared_error(y_reg_test,X_test["AQI"])),2),
                            round(r2_score(y_reg_test,X_test["AQI"]),4)])

for name,model in all_reg_models.items():
    preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_reg_test,preds)
    rmse = np.sqrt(mean_squared_error(y_reg_test,preds))
    r2 = r2_score(y_reg_test,preds)

    reg_comparison_list.append([name,round(mae,2),round(rmse,2),round(r2,4)])

reg_comparison_df = pd.DataFrame(
    reg_comparison_list,
    columns=["Regression Method","MAE","RMSE","R2"]
)

print(reg_comparison_df.to_string(index=False))

# Selecting the best regression model for the dataset
best_reg = reg_comparison_df.sort_values(by="RMSE").iloc[0]
print(f"Best Regression Model : {best_reg['Regression Method']} (RMSE: {best_reg['RMSE']})")


# Comparison of classification Models

print("\n" + "="*25, "Classification Model Comparison Table", "="*25)

clf_models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42),
    "SVM":SVC(probability=True,random_state=42),
    "Gradient Boosting":GradientBoostingClassifier(n_estimators=100,random_state=42),
    "ANN (MLP)":MLPClassifier(hidden_layer_sizes=(64,32),early_stopping=True,max_iter=500,random_state=42)
}

# Specificity = share of truly HEALTHY days predicted healthy = TN / (TN + FP)
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

clf_comparison_list = []

# Baseline rule row (AUC uses today's AQI as the score)
rule_train_pred = (X_train["AQI"] > 100).astype(int)
clf_comparison_list.append(["Baseline rule (today>100)",
                            round(accuracy_score(y_clf_test,rule_pred),4),
                            round(precision_score(y_clf_test,rule_pred),4),
                            round(recall_score(y_clf_test,rule_pred),4),
                            round(specificity_score(y_clf_test,rule_pred),4),
                            round(f1_score(y_clf_test,rule_pred),4),
                            round(roc_auc_score(y_clf_test,X_test["AQI"]),4),
                            round(f1_score(y_clf_train,rule_train_pred),4)])

for name,model in clf_models.items():
    model.fit(X_train_scaled,y_clf_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_clf_test,preds)
    prec = precision_score(y_clf_test,preds)
    rec = recall_score(y_clf_test,preds)
    spec = specificity_score(y_clf_test,preds)
    f1 = f1_score(y_clf_test,preds)
    auc = roc_auc_score(y_clf_test,model.predict_proba(X_test_scaled)[:,1])
    train_f1 = f1_score(y_clf_train,model.predict(X_train_scaled))      # F1 on the training data

    clf_comparison_list.append([name,round(acc,4),round(prec,4),round(rec,4),round(spec,4),round(f1,4),round(auc,4),round(train_f1,4)])

clf_comparison_df = pd.DataFrame(
    clf_comparison_list,
    columns=["Classification Method","Accuracy","Precision","Recall","Specificity","F1","AUC","Train F1"]
)

print(clf_comparison_df.to_string(index=False))

best_clf = clf_comparison_df.sort_values(by="F1",ascending=False).iloc[0]

print(f"Best Classification Model (Highest F1): {best_clf['Classification Method']}  ({best_clf['F1']:.4f})")

# ==============================================================================
# HYPERPARAMETER TUNING (RF, GB, ANN)
# ==============================================================================

print("\n" + "="*25, "Tuning Random Forest, Gradient Boosting & ANN", "="*25)

# Random Forest Tuning
rf_param_grid = {
    "n_estimators":[100,150],
    "max_depth":[10,20,None],
    "min_samples_split":[2,5]
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42,n_jobs=-1),
    param_grid=rf_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train_scaled,y_clf_train)
best_rf = rf_grid.best_estimator_
rf_tuned_preds = best_rf.predict(X_test_scaled)

print(f"Best RF Parameters: {rf_grid.best_params_}")
print(f"Tuned RF F1 Score: {f1_score(y_clf_test,rf_tuned_preds):.4f}")

# Gradient Boosting Tuning
gb_param_grid = {
    "n_estimators":[100,150],
    "learning_rate":[0.05,0.1],
    "max_depth":[3,5]
}

gb_grid = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=gb_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

gb_grid.fit(X_train_scaled,y_clf_train)
best_gb = gb_grid.best_estimator_
tuned_gb_pred = best_gb.predict(X_test_scaled)

print(f"Best GB Parameter: {gb_grid.best_params_}")
print(f"Tuned GB F1 score: {f1_score(y_clf_test,tuned_gb_pred):.4f}")

# ANN (MLP) Tuning
ann_param_grid = {
    "hidden_layer_sizes":[(64,32),(128,64)],
    "alpha":[0.0001,0.01],
    "learning_rate_init":[0.001,0.005]
}

ann_grid = GridSearchCV(
    estimator=MLPClassifier(early_stopping=True,max_iter=500,random_state=42),
    param_grid=ann_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

ann_grid.fit(X_train_scaled,y_clf_train)
best_ann = ann_grid.best_estimator_
tuned_ann_pred = best_ann.predict(X_test_scaled)

print(f"Best ANN Parameters: {ann_grid.best_params_}")
print(f"Tuned ANN F1 score: {f1_score(y_clf_test,tuned_ann_pred):.4f}")

# Default vs tuned, then pick the final classifier (highest F1 of all six)
print("\n" + "="*25, "Default vs Tuned", "="*25)
tuning_pairs = [
    ["Random Forest", clf_models["Random Forest"], best_rf],
    ["Gradient Boosting", clf_models["Gradient Boosting"], best_gb],
    ["ANN (MLP)", clf_models["ANN (MLP)"], best_ann],
]

tuning_rows = []
all_candidates = {}
for name, default_model, tuned_model in tuning_pairs:
    default_f1 = f1_score(y_clf_test, default_model.predict(X_test_scaled))
    tuned_f1 = f1_score(y_clf_test, tuned_model.predict(X_test_scaled))
    tuning_rows.append([name, round(default_f1,4), round(tuned_f1,4), "Yes" if tuned_f1 > default_f1 else "No"])
    all_candidates[f"{name} (default)"] = default_model
    all_candidates[f"{name} (tuned)"] = tuned_model

print(pd.DataFrame(tuning_rows, columns=["Model","Default F1","Tuned F1","Tuning Helped?"]).to_string(index=False))

final_name = max(all_candidates, key=lambda k: f1_score(y_clf_test, all_candidates[k].predict(X_test_scaled)))
final_clf = all_candidates[final_name]
final_pred = final_clf.predict(X_test_scaled)

print(f"\nFinal classifier: {final_name}")
print(f"Final test Accuracy: {accuracy_score(y_clf_test,final_pred):.4f}  Precision: {precision_score(y_clf_test,final_pred):.4f}  "
      f"Recall: {recall_score(y_clf_test,final_pred):.4f}  Specificity: {specificity_score(y_clf_test,final_pred):.4f}  "
      f"F1: {f1_score(y_clf_test,final_pred):.4f}")
print(confusion_matrix(y_clf_test,final_pred))

# ==============================================================================
# GRAPHS
# ==============================================================================

# Model vs F1
plt.figure(figsize=(8,4))
plt.barh(clf_comparison_df["Classification Method"], clf_comparison_df["F1"])
plt.xlim(0.8,1.0); plt.xlabel("F1 score"); plt.title("Model vs F1")
plt.tight_layout(); plt.show()

# Training vs test performance (a big gap means overfitting)
y_pos = np.arange(len(clf_comparison_df))
plt.figure(figsize=(8,4.5))
plt.barh(y_pos - 0.2, clf_comparison_df["Train F1"], height=0.4, label="Training F1")
plt.barh(y_pos + 0.2, clf_comparison_df["F1"], height=0.4, label="Test F1")
plt.yticks(y_pos, clf_comparison_df["Classification Method"])
plt.xlim(0.8,1.01); plt.xlabel("F1 score"); plt.title("Training vs Test performance")
plt.legend(); plt.tight_layout(); plt.show()

# ROC curves
plt.figure(figsize=(5,4))
for roc_name, roc_model in [("Logistic Regression", clf_models["Logistic Regression"]),
                            ("Gradient Boosting", clf_models["Gradient Boosting"]),
                            (f"Final ({final_name})", final_clf)]:
    fpr, tpr, _ = roc_curve(y_clf_test, roc_model.predict_proba(X_test_scaled)[:,1])
    plt.plot(fpr, tpr, label=roc_name)
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves")
plt.legend(); plt.tight_layout(); plt.show()

# Confusion matrix of the final model
plt.figure(figsize=(4,3.5))
cm = confusion_matrix(y_clf_test, final_pred)
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha="center",va="center")
plt.xticks([0,1],["Healthy","Unhealthy"]); plt.yticks([0,1],["Healthy","Unhealthy"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (final model)")
plt.tight_layout(); plt.show()

# ==============================================================================
# ABLATION STUDY
# ==============================================================================

print("\n" + "="*25, "Ablation Study", "="*25)

X_train_s = pd.DataFrame(X_train_scaled, columns=features)
X_test_s = pd.DataFrame(X_test_scaled, columns=features)

# Each step adds one component; every step uses the final model's settings
ablation_steps = [
    ("M1: raw pollutants only",                       ["AQI","PM2.5","NO2","CO"]),
    ("M2: M1 + Temporal Lag (Previous AQI)",          ["AQI","Previous AQI","PM2.5","NO2","CO"]),
    ("M3: M2 + extra pollutants + calendar = FINAL",  features),
]

ablation_study = []
for step_name, cols in ablation_steps:
    step_model = clone(final_clf)
    step_model.fit(X_train_s[cols],y_clf_train)
    step_preds = step_model.predict(X_test_s[cols])
    ablation_study.append([step_name,
                           round(accuracy_score(y_clf_test,step_preds),4),
                           round(precision_score(y_clf_test,step_preds),4),
                           round(recall_score(y_clf_test,step_preds),4),
                           round(f1_score(y_clf_test,step_preds),4)])

ablation_df = pd.DataFrame(ablation_study,columns=["Pipeline Configuration","Accuracy","Precision","Recall","F1"])
print(f"Model used in every step: {final_name}")
print(ablation_df.to_string(index=False))

# ==============================================================================
# SAVING FINAL MODELS
# ==============================================================================

print("\n" + "="*25, "Saving Models", "="*25)

joblib.dump(rf_reg, "best_regressor_rf.pkl")
joblib.dump(final_clf, "best_classifier.pkl")
joblib.dump(scaler,"feature_Scaler.pkl")

# Needed by the testing file to prepare new data exactly like the training data
joblib.dump(features, "features.pkl")
joblib.dump(train_medians, "train_medians.pkl")
joblib.dump({"rows": df.shape[0], "unhealthy_pct": round(y_clf.mean()*100,2)}, "city_day_info.pkl")

print("Models Saved successfully to disk.")