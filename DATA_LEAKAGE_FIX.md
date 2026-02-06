# Data Leakage Fix Documentation

## 1. Problem Identified

**Lag features present.** The earlier dataset included lag/rolling-derived predictors, including prior-year life expectancy and related rolling or year-over-year transformations. These features embed future information relative to the temporal split because a target-adjacent statistic from neighboring years can leak signal across the train/test boundary.

**Why they caused data leakage.** The modeling setup uses a temporal split (train: years ≤ 2013; test: years ≥ 2014). Lag or rolling features constructed over time can inadvertently incorporate information from years that appear in the test set when computed on the full dataset, thereby leaking test-period information into training features. This inflates predictive performance because the model learns from data that should be unseen at training time.

**Why R² ≈ 0.99 was suspicious.** For socio-economic and health outcome prediction, near-perfect generalization (Test R² ≈ 0.99) is implausible given measurement noise, reporting variation, and country-level heterogeneity. Such values are consistent with leakage-driven overfitting rather than genuine predictive capacity.

## 2. Solution Implemented

**Removed features.** All lag-based and rolling window features were removed from the modeling dataset. The lag-free dataset used is `clean_dataset_no_lags.csv`.

**Temporal split preserved.** The temporal split was retained (train: years ≤ 2013; test: years ≥ 2014) to reflect real-world forecasting where future years must be predicted from past years only.

**What the new approach tests.** The revised pipeline evaluates how well contemporaneous country-level indicators explain future life expectancy without access to future-derived signals. This provides a more honest estimate of out-of-sample performance.

## 3. Results After Fix

### Old vs New Test R²

| Model             |   Old Test R2 |   New Test R2 |   R2 Drop |
|:------------------|--------------:|--------------:|----------:|
| Random Forest     |         0.992 |      0.93771  |  0.05429  |
| Gradient Boosting |         0.989 |      0.931245 |  0.057755 |
| Ridge             |         0.987 |      0.807235 |  0.179765 |
| Linear Regression |         0.985 |      0.806321 |  0.178679 |

### Interpretation

The drop in Test R² is expected and desirable because it indicates removal of leakage. The models now generalize based on legitimate predictors rather than artifacts that inadvertently carried target information across time.

### New Performance Metrics (Lag-Free)

| Model             |   Train R2 |   Test RMSE |   Test MAE |
|:------------------|-----------:|------------:|-----------:|
| Random Forest     |   0.994968 |     2.07711 |    1.44968 |
| Gradient Boosting |   0.987734 |     2.18223 |    1.5907  |
| Ridge             |   0.819577 |     3.65395 |    2.78212 |
| Linear Regression |   0.819729 |     3.6626  |    2.7916  |
