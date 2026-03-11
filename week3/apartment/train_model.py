"""
Apartment Price Prediction - Model Training Script
Iterative modeling process with documentation for each iteration.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from math import radians, cos, sin, asin, sqrt

# ============================================================
# 1. Load and Prepare Data
# ============================================================

# Load enriched apartment data from week2
df = pd.read_csv(
    "../../week2/apartments_data_enriched_with_new_features.csv",
    sep=",",
    encoding="utf-8",
)

# Load BFS municipality data
df_bfs = pd.read_csv("bfs_municipality_and_tax_data.csv", sep=",", encoding="utf-8")
df_bfs["tax_income"] = df_bfs["tax_income"].str.replace("'", "").astype(float)

print(f"Dataset: {len(df)} apartments")
print(f"Columns: {list(df.columns)}")

# ============================================================
# 2. Data Cleaning
# ============================================================

# Fix tax_income if it's string
if df["tax_income"].dtype == object:
    df["tax_income"] = df["tax_income"].str.replace("'", "").astype(float)

# Drop rows with missing essential values
df = df.dropna(subset=["rooms", "area", "price", "pop", "pop_dens", "frg_pct", "emp", "tax_income", "lat", "lon"])

# Remove obvious outliers
df = df[(df["price"] > 200) & (df["price"] < 15000)]
df = df[(df["area"] > 10) & (df["area"] < 500)]
df = df[df["rooms"] > 0]

print(f"After cleaning: {len(df)} apartments")

# ============================================================
# 3. Feature Engineering - NEW FEATURE: distance_to_zurich
# ============================================================

# Zurich main station coordinates
ZH_LAT = 47.3769
ZH_LON = 8.5417


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points using haversine formula."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


# Compute distance to Zurich center for each apartment
df["distance_to_zurich"] = df.apply(
    lambda row: haversine(row["lat"], row["lon"], ZH_LAT, ZH_LON), axis=1
)

# Also compute area_per_room as derived feature
df["area_per_room"] = df["area"] / df["rooms"]

# Compute mean distance per BFS municipality for the app lookup
dist_per_bfs = df.groupby("bfs_number")["distance_to_zurich"].mean().reset_index()
dist_per_bfs.columns = ["bfs_number", "distance_to_zurich"]

# Merge distance into BFS data for the app
df_bfs_enriched = df_bfs.merge(dist_per_bfs, on="bfs_number", how="left")
# Fill missing distances with median
median_dist = df_bfs_enriched["distance_to_zurich"].median()
df_bfs_enriched["distance_to_zurich"] = df_bfs_enriched["distance_to_zurich"].fillna(median_dist)

# Save enriched BFS data for the app
df_bfs_enriched.to_csv("bfs_municipality_data_enriched.csv", index=False)
print("Saved enriched BFS data with distance_to_zurich")
print(f"Distance range: {df['distance_to_zurich'].min():.1f} - {df['distance_to_zurich'].max():.1f} km")

# ============================================================
# ITERATION 1: Baseline Models
# ============================================================

print("\n" + "=" * 60)
print("ITERATION 1: Baseline Models")
print("=" * 60)

# Baseline features (same as original model)
baseline_features = ["rooms", "area", "pop", "pop_dens", "frg_pct", "emp", "tax_income"]
X_baseline = df[baseline_features].values
y = df["price"].values

# Model 1a: Random Forest (default params)
rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
scores_rf1 = cross_val_score(rf_baseline, X_baseline, y, cv=5, scoring="r2")

# Model 1b: Ridge Regression
ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
scores_ridge1 = cross_val_score(ridge_pipe, X_baseline, y, cv=5, scoring="r2")

# Model 1c: Gradient Boosting
gb_baseline = GradientBoostingRegressor(n_estimators=100, random_state=42)
scores_gb1 = cross_val_score(gb_baseline, X_baseline, y, cv=5, scoring="r2")

print(f"\nBaseline features: {baseline_features}")
print(f"Random Forest:      R² = {scores_rf1.mean():.4f} (±{scores_rf1.std():.4f})")
print(f"Ridge Regression:   R² = {scores_ridge1.mean():.4f} (±{scores_ridge1.std():.4f})")
print(f"Gradient Boosting:  R² = {scores_gb1.mean():.4f} (±{scores_gb1.std():.4f})")

# ============================================================
# ITERATION 2: Enhanced Models with New Features
# ============================================================

print("\n" + "=" * 60)
print("ITERATION 2: Enhanced Models with New Features")
print("=" * 60)

# Enhanced features including new distance_to_zurich and area_per_room
enhanced_features = [
    "rooms", "area", "pop", "pop_dens", "frg_pct", "emp", "tax_income",
    "distance_to_zurich", "area_per_room",
]
X_enhanced = df[enhanced_features].values

# Model 2a: Tuned Random Forest
rf_tuned = RandomForestRegressor(
    n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42
)
scores_rf2 = cross_val_score(rf_tuned, X_enhanced, y, cv=5, scoring="r2")

# Model 2b: Ridge with enhanced features
ridge_pipe2 = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
scores_ridge2 = cross_val_score(ridge_pipe2, X_enhanced, y, cv=5, scoring="r2")

# Model 2c: Tuned Gradient Boosting
gb_tuned = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.1, min_samples_split=5, random_state=42
)
scores_gb2 = cross_val_score(gb_tuned, X_enhanced, y, cv=5, scoring="r2")

# Model 2d: Lasso
lasso_pipe = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=1.0))])
scores_lasso2 = cross_val_score(lasso_pipe, X_enhanced, y, cv=5, scoring="r2")

print(f"\nEnhanced features: {enhanced_features}")
print(f"Tuned Random Forest:     R² = {scores_rf2.mean():.4f} (±{scores_rf2.std():.4f})")
print(f"Ridge Regression:        R² = {scores_ridge2.mean():.4f} (±{scores_ridge2.std():.4f})")
print(f"Tuned Gradient Boosting: R² = {scores_gb2.mean():.4f} (±{scores_gb2.std():.4f})")
print(f"Lasso Regression:        R² = {scores_lasso2.mean():.4f} (±{scores_lasso2.std():.4f})")

# ============================================================
# ITERATION 3: Best model with log-transformed target
# ============================================================

print("\n" + "=" * 60)
print("ITERATION 3: Log-transformed target + best model tuning")
print("=" * 60)

y_log = np.log1p(y)

# Model 3a: Tuned Random Forest with log target
rf_log = RandomForestRegressor(
    n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42
)
scores_rf3 = cross_val_score(rf_log, X_enhanced, y_log, cv=5, scoring="r2")

# Model 3b: Gradient Boosting with log target
gb_log = GradientBoostingRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05, min_samples_split=5, random_state=42
)
scores_gb3 = cross_val_score(gb_log, X_enhanced, y_log, cv=5, scoring="r2")

print(f"\nLog-transformed target with enhanced features:")
print(f"Random Forest (log):      R² = {scores_rf3.mean():.4f} (±{scores_rf3.std():.4f})")
print(f"Gradient Boosting (log):  R² = {scores_gb3.mean():.4f} (±{scores_gb3.std():.4f})")
print("(Note: R² on log-scale; actual predictions use expm1 to convert back)")

# ============================================================
# Select and Save Best Model
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL SELECTION")
print("=" * 60)

# Compare all models
results = {
    "RF Baseline": scores_rf1.mean(),
    "Ridge Baseline": scores_ridge1.mean(),
    "GB Baseline": scores_gb1.mean(),
    "RF Tuned (enhanced)": scores_rf2.mean(),
    "Ridge (enhanced)": scores_ridge2.mean(),
    "GB Tuned (enhanced)": scores_gb2.mean(),
    "Lasso (enhanced)": scores_lasso2.mean(),
    "RF Log (enhanced)": scores_rf3.mean(),
    "GB Log (enhanced)": scores_gb3.mean(),
}

print("\nAll model R² scores:")
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:30s} {score:.4f}")

# Train the best non-log model on full data (easier to interpret predictions)
best_name = max(
    {k: v for k, v in results.items() if "Log" not in k},
    key=lambda k: results[k],
)
print(f"\nBest model (non-log): {best_name} with R² = {results[best_name]:.4f}")

# Check if log models are substantially better
best_log_name = max(
    {k: v for k, v in results.items() if "Log" in k},
    key=lambda k: results[k],
)
print(f"Best model (log):     {best_log_name} with R² = {results[best_log_name]:.4f}")

# Train final model: use the RF with log-transformed target (best overall)
print(f"\nTraining final model: Random Forest with log-transformed target + enhanced features...")
final_model = RandomForestRegressor(
    n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42
)
final_model.fit(X_enhanced, y_log)
# Note: predictions need np.expm1() to convert back to CHF

# Save model
model_path = "apartment_price_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
print(f"Model saved to {model_path}")

# Feature importances
print("\nFeature Importances:")
for feat, imp in sorted(
    zip(enhanced_features, final_model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"  {feat:25s} {imp:.4f}")

# Save feature list for the app
with open("model_features.pkl", "wb") as f:
    pickle.dump(enhanced_features, f)

print("\nDone! Files created:")
print(f"  - {model_path} (trained model)")
print(f"  - model_features.pkl (feature list)")
print(f"  - bfs_municipality_data_enriched.csv (BFS data with distance_to_zurich)")
