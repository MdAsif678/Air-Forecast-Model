# Air Quality Forecasting & Early Warning System

A machine learning project that uses today's pollution readings to predict **whether tomorrow's air will be unhealthy (AQI > 100)** and **tomorrow's exact AQI** for Indian cities.

Air quality apps show today's AQI. This project aims to give a one-day warning, so people, schools and hospitals can prepare before the air turns bad.

## Results

All scores are on a 20% held-out test set (4,408 days).

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Baseline rule (no ML: unhealthy if today's AQI > 100) | 0.8761 | 0.8969 | 0.9398 |
| Logistic Regression | 0.8814 | 0.9004 | 0.9504 |
| k-NN | 0.8736 | 0.8941 | 0.9338 |
| Decision Tree | 0.8505 | 0.8763 | 0.8434 |
| **Random Forest (final)** | **0.8963** | **0.9140** | **0.9619** |
| SVM | 0.8879 | 0.9064 | 0.9509 |
| Gradient Boosting | 0.8934 | 0.9113 | 0.9599 |
| ANN (MLP) | 0.8907 | 0.9089 | 0.9597 |

- **Regression (next-day AQI):** Random Forest RMSE 32.96 vs 41.96 for "tomorrow = today".
- **External test** on 74,402 unseen station-level readings: F1 0.9385 vs 0.9184 for the baseline, RMSE 36.29 vs 47.22.
- **Ablation:** raw pollutants F1 0.9015 → + yesterday's AQI 0.9036 → + extra pollutants and calendar features 0.9140. Feature engineering helped more than hyperparameter tuning, which did not improve test F1.

## Approach

1. **EDA:** missing values, IQR outliers, correlation heatmap. AQI values above 500 are impossible on the CPCB scale and are removed.
2. **Feature engineering:** `Previous AQI` (yesterday's AQI per city), `Month`, `DayOfWeek`, plus PM10, NO, NOx, SO2 and O3. Only consecutive days are kept.
3. **No data leakage:** missing values are filled with training-set medians, and the scaler is fitted on training data only.
4. **Models:** Logistic/Linear Regression, k-NN, Decision Tree, Random Forest, SVM, Gradient Boosting and an ANN (MLP), compared against the baseline rule.
5. **Tuning:** GridSearchCV with 5-fold cross-validation on Random Forest, Gradient Boosting and the ANN.
6. **Validation:** the saved model, scaler and medians are reused, unchanged, on the station-level dataset.

## Dataset

[Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) on Kaggle, compiled from CPCB data. The CSVs are not included in this repo.

Download it and put `city_day.csv` and `station_day.csv` in the project folder.

## How to run

```bash
pip install -r requirements.txt
python aqi_main.py            # EDA, training, comparison, tuning, graphs, ablation; saves the models (.pkl)
python validation_station.py  # tests the saved models on station_day.csv
```

`aqi_main.py` opens each graph in a window; close it to continue.

## Limitations

- No weather data (wind, rain, temperature). Most errors happen when AQI is near 100.
- Random train-test split. A chronological split would test forecasting more strictly.
- Both datasets come from the same CPCB source, so the external test is related, not fully independent.
- The final model was chosen by test F1.
