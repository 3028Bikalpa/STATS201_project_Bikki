- [Life Expectancy Forecasting (Time-Series + ML)](#life-expectancy-forecasting-time-series--ml)
  - [Project Goal](#project-goal)
  - [Dataset](#dataset)
  - [Notebook 1 — `data_cleaning.ipynb` (Cleaning + Feature Engineering)](#notebook-1--data_cleaningipynb-cleaning--feature-engineering)
    - [What happens in this notebook](#what-happens-in-this-notebook)
    - [Key numbers (from the notebook)](#key-numbers-from-the-notebook)
    - [Visual outputs to include](#visual-outputs-to-include)
  - [Notebook 2 — `train_models.ipynb` (Model Training + Evaluation)](#notebook-2--train_modelsipynb-model-training--evaluation)
    - [What happens in this notebook](#what-happens-in-this-notebook-1)
  - [Train/Test Split Summary (Temporal)](#traintest-split-summary-temporal)
  - [Model Results (Most Important Numbers)](#model-results-most-important-numbers)
    - [Baselines (Week 3)](#baselines-week-3)
    - [Advanced Models + Feature-Set Comparison (Week 4)](#advanced-models--feature-set-comparison-week-4)
  - [What Drives Predictions? (Feature Importance Highlights)](#what-drives-predictions-feature-importance-highlights)
# Life Expectancy Forecasting (Time-Series + ML)

This project builds an end-to-end workflow to **clean and prepare the WHO Life Expectancy dataset**, engineer **time-series features** (lags, rolling means, YoY change), and **train/evaluate multiple regression models** to predict **Life Expectancy (years)**.

The repository is organized around two Jupyter notebooks:

- **`data_cleaning.ipynb`** → data loading, quality checks, cleaning decisions, EDA, and feature engineering
- **`train_models.ipynb`** → temporal split, scaling, baselines, advanced models, comparisons, and interpretation

---

## Project Goal

Predict **Life Expectancy** using country-level health/economic indicators and **past life expectancy trends** (time-series features), while avoiding time leakage by using a **temporal train/test split**.

---

## Dataset

- Observations represent **Country-Year** records with health + socio-economic indicators.
- Target: **`Life expectancy`** (years)

> Place the raw CSV in a local folder (recommended: `data/`) and update the file paths inside the notebooks if needed.

---

## Notebook 1 — `data_cleaning.ipynb` (Cleaning + Feature Engineering)

### What happens in this notebook

1. **Load + initial exploration**
2. **Missing value audit** (counts + percentages)
3. **Cleaning decisions**
   - Remove rows where the target is missing
   - Check duplicates by `Country` + `Year`
   - Outlier detection on target via conservative IQR bounds (3×IQR)
   - Encode `Status` → binary feature
4. **Time-series feature engineering (per country)**
   - `Years_Since_2000`
   - `Life_Expectancy_Lag_1`, `Lag_2`, `Lag_3`
   - `Life_Expectancy_RollingMean_3yr`, `RollingMean_5yr` (past-only)
   - `Life_Expectancy_YoY_Change` (past-only)
5. **EDA**
   - target distribution + comparisons
   - correlations with the target
   - scatter plots for top correlated predictors

### Key numbers (from the notebook)

**Raw dataset**
- Rows: **2938**
- Columns: **22**

**Missingness highlights**
- Target missing (`Life expectancy`): **10 rows (0.34%)**
- Largest missing predictors:
  - `Population`: **652 (22.19%)**
  - `Hepatitis B`: **553 (18.82%)**
  - `GDP`: **448 (15.25%)**

**After cleaning**
- Removed missing target rows: **10**
- Remaining rows: **2928**
- Duplicate `Country-Year` pairs: **0**
- Target outliers detected: **0**
  - Target range: **36.3 → 89.0**
  - Outlier bounds: **25.3 → 113.5** (3×IQR)

**Status distribution**
- Developing: **2416**
- Developed: **512**

**Feature engineering impact**
- Before time-series features: **(2928, 23)**
- After time-series features: **(2928, 30)** → **+6 new features**
- Expected missingness introduced by lags:
  - `Lag_1`: **183 (6.2%)**
  - `Lag_2`: **366 (12.5%)**
  - `Lag_3`: **549 (18.8%)**
  - `YoY_Change`: **184 (6.3%)**

**Important note on leakage prevention**
- This notebook **does not impute** missing predictor values.
- Imputation is deferred to model training to avoid temporal leakage.

### Visual outputs to include

- **Target distribution (histogram + boxplot)**
![alt text](images/life_expectancy_distribution.png)
- **Scatter plots of top correlated features**
![alt text](images/scatter_plots_top_features.png)

---

## Notebook 2 — `train_models.ipynb` (Model Training + Evaluation)

### What happens in this notebook

1. **Load the cleaned dataset**
2. **Feature selection**
   - Excludes: `Country`, `Year`, `Status`, and the target
   - Uses engineered time-series features (lags/rolling/YoY) + `Years_Since_2000` + health/econ predictors
3. **Temporal split (time-series forecasting logic)**
   - Train years: **2003–2013**
   - Test years: **2014–2015**
   - To ensure valid lag features, rows missing time-series features are removed
4. **Scaling**
   - `StandardScaler` fit on training only
5. **Baseline models**
   - DummyRegressor (mean prediction sanity check)
   - Linear Regression
   - Decision Tree Regressor (constrained to reduce overfitting)
6. **Advanced models + controlled comparisons (Week 4)**
   - Random Forest
   - Gradient Boosting
   - Ridge Regression
   - Lasso Regression
   - Each tested across multiple feature sets (baseline / engineered / polynomial)
7. **Diagnostics**
   - Predicted vs actual plots
   - Residual analysis
   - Model behavior comparison (prediction correlation matrix)
   - Feature importance comparison across models

---

## Train/Test Split Summary (Temporal)

From the notebook:

- Original feature matrix: **(2928, 26 features)**
- Rows with complete time-series features: **2379 (81.2%)**
- Removed due to missing lag features: **549**

**Final modeling dataset**
- **Train:** **2013 samples** (years **2003–2013**)
- **Test:** **366 samples** (years **2014–2015**)

---

## Model Results (Most Important Numbers)

### Baselines (Week 3)

| Model | Test RMSE (years) | Test MAE (years) | Test R² |
|------|--------------------|------------------|--------:|
| DummyRegressor (mean) | **8.6066** | **7.3219** | **-0.0695** |
| Linear Regression | **1.5912** | **0.8281** | **0.9634** |
| Decision Tree (max_depth=10) | **1.2865** | **0.6727** | **0.9761** |

**Interpretation:** Even a simple linear model performs strongly, and the decision tree improves further.

---

### Advanced Models + Feature-Set Comparison (Week 4)

The notebook runs controlled experiments across **Baseline / Engineered / Polynomial** feature sets.

**Best overall configuration (by Test R²):**
- **Gradient Boosting (Baseline features)**
  - **Test R²:** **0.9821**
  - **Test RMSE:** **1.1148 years**
  - **Test MAE:** **0.5630 years**

Selected runner-ups:
- **Random Forest (Engineered features)** → Test R² **0.9813**, RMSE **1.1369**
- **Random Forest (Baseline features)** → Test R² **0.9813**, RMSE **1.1377**

---

## What Drives Predictions? (Feature Importance Highlights)

Across models, the strongest predictors consistently include the **time-series life expectancy features**, especially:

- `Life_Expectancy_RollingMean_3yr`
- `Life_Expectancy_Lag_1`
- `Life_Expectancy_Lag_2`
- `Life_Expectancy_RollingMean_5yr`
- `Life_Expectancy_YoY_Change`
- plus health burden indicators (e.g., mortality measures) depending on the model

![alt text](images/week4_feature_importance_comparison.png)

---
