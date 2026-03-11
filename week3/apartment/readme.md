---
title: Zurich Apartment Price Predictor
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Apartment Price Prediction - Model Iterations Documentation
## Task: Apartment Rental Price Prediction (Regression)

---

## Overview

This project predicts monthly rental prices for apartments in the Canton of Zurich using a Random Forest regressor. The model was developed iteratively over three iterations, progressively adding new features and tuning hyperparameters. The final model achieves an R² of 0.66 using 5-fold cross-validation.

---

## Summary of Iterative Process

| Iteration | Objective | Key Changes | Models Used | CV Mean R² | CV Std Dev | Change in Performance | Fit Diagnosis |
|-----------|-----------|-------------|-------------|------------|------------|-----------------------|---------------|
| **1** | Build baseline model | - Outlier removal (price, area)<br>- Drop missing values<br>- 7 original features<br>- 5-fold CV | Random Forest (n_estimators=100)<br>Ridge Regression (alpha=1.0)<br>Gradient Boosting (n_estimators=100) | 0.54 (Ridge)<br>0.51 (GB)<br>0.46 (RF) | 0.09<br>0.06<br>0.05 | Baseline | ☑ Underfitting |
| **2** | Add new features + tuning | - Added **distance_to_zurich** (haversine distance to Zurich HB)<br>- Added **area_per_room** (derived feature)<br>- Hyperparameter tuning<br>- 5-fold CV | Tuned RF (n_estimators=300, max_depth=20)<br>Ridge (alpha=1.0)<br>Tuned GB (n_estimators=300, max_depth=5)<br>Lasso (alpha=1.0) | 0.59 (RF)<br>0.53 (Lasso)<br>0.53 (Ridge)<br>0.52 (GB) | 0.06<br>0.08<br>0.08<br>0.08 | +0.13 (RF) | ☐ Overfitting ☐ Underfitting ☑ Good Fit |
| **3** | Log-transform target | - Log1p transformation of target variable<br>- Enhanced features from Iter 2<br>- Increased n_estimators to 500<br>- 5-fold CV | RF (n_estimators=500, max_depth=20)<br>GB (n_estimators=500, max_depth=5, lr=0.05) | 0.66 (RF)<br>0.61 (GB) | 0.05<br>0.06 | +0.07 (RF) | ☐ Overfitting ☐ Underfitting ☑ Good Fit |

---

## Preprocessing Steps

1. **Data loading**: Loaded 804 apartment listings from enriched dataset (week 2), merged with BFS municipality demographic data
2. **Missing value handling**: Dropped rows missing essential features (rooms, area, price, population stats, coordinates)
3. **Outlier removal**: Removed apartments with price < 200 or > 15,000 CHF, area < 10 or > 500 m², rooms <= 0
4. **Feature engineering**:
   - **distance_to_zurich** (NEW): Haversine distance in km from apartment to Zurich main station (47.3769°N, 8.5417°E). Computed per listing, then averaged per municipality for the prediction app.
   - **area_per_room**: Living area divided by number of rooms
5. **Target transformation** (Iteration 3): Applied `log1p` to the rental price for training; predictions are converted back with `expm1`
6. **String cleaning**: Converted `tax_income` from formatted strings (e.g., "108'788") to float

---

## Features Used (9 total)

| Feature | Source | Description |
|---------|--------|-------------|
| rooms | User input | Number of rooms |
| area | User input | Living area in m² |
| pop | BFS data | Municipality population |
| pop_dens | BFS data | Population density |
| frg_pct | BFS data | Foreign resident percentage |
| emp | BFS data | Number of employed persons |
| tax_income | BFS data | Average taxable income |
| distance_to_zurich | **Engineered (NEW)** | Haversine distance to Zurich HB in km |
| area_per_room | **Engineered** | area / rooms |

---

## Feature Importances (Final Model)

| Feature | Importance |
|---------|-----------|
| area | 0.4679 |
| distance_to_zurich | 0.2777 |
| rooms | 0.1148 |
| area_per_room | 0.0488 |
| pop_dens | 0.0246 |
| tax_income | 0.0242 |
| pop | 0.0233 |
| frg_pct | 0.0100 |
| emp | 0.0087 |

The newly engineered **distance_to_zurich** is the second most important feature (28%), confirming that proximity to the city center is a strong predictor of rental prices.

---

## Evaluation Method

- **Metric**: R² (coefficient of determination)
- **Validation**: 5-fold cross-validation on the full dataset
- All models evaluated with the same CV splits for fair comparison

---

## Final Selected Model

- **Model**: Random Forest Regressor
- **Hyperparameters**: n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42
- **Target transformation**: log1p (predictions converted back with expm1)
- **CV R²**: 0.6613 (±0.0501)
- **Reason for selection**: Highest cross-validation R² across all iterations and models. The log transformation improved performance by 0.07 R² over the non-log version.

---

## Application

The Gradio web app (`app.py`) accepts:
- Number of rooms
- Living area (m²)
- Town (dropdown of 100+ Zurich-area municipalities)

It looks up municipality data from the enriched BFS dataset, computes `area_per_room`, and returns the predicted monthly rent in CHF.

---

## Files

| File | Description |
|------|-------------|
| `train_model.py` | Full training script with all 3 iterations |
| `app.py` | Gradio web application |
| `apartment_price_model.pkl` | Trained Random Forest model |
| `model_features.pkl` | Ordered feature list |
| `bfs_municipality_data_enriched.csv` | BFS data with distance_to_zurich |
| `bfs_municipality_and_tax_data.csv` | Original BFS municipality data |
| `requirements.txt` | Python dependencies |
