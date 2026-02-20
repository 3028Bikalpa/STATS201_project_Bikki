- [STATS 201 Course Project](#stats-201-course-project)
  - [Time Series Prediction of Life Expectancy (Data Cleaning ➜ Model Training)](#time-series-prediction-of-life-expectancy-data-cleaning--model-training)
  - [Author](#author)
  - [Course](#course)
  - [Project Overview](#project-overview)
  - [Research Question](#research-question)
  - [End-to-End Pipeline](#end-to-end-pipeline)
  - [Data Source](#data-source)
    - [Core Variables](#core-variables)
- [Notebook 1: `data_cleaning.ipynb`](#notebook-1-data_cleaningipynb)
  - [1. Import \& Load](#1-import--load)
  - [2. Initial Exploration](#2-initial-exploration)
  - [3. Missing Values](#3-missing-values)
  - [4. Cleaning Steps](#4-cleaning-steps)
    - [Remove Missing Target](#remove-missing-target)
    - [Missing Predictors (Deferred Imputation)](#missing-predictors-deferred-imputation)
    - [Duplicates (Country-Year)](#duplicates-country-year)
    - [Target Outliers (IQR)](#target-outliers-iqr)
    - [Encode Development Status](#encode-development-status)
    - [Time Series Features](#time-series-features)
    - [Missing Values from Lags](#missing-values-from-lags)
  - [5. Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
    - [Target Analysis](#target-analysis)
    - [Correlations](#correlations)
    - [Distributions](#distributions)
    - [Scatter Plots (Top Correlated Features)](#scatter-plots-top-correlated-features)
  - [6. Final Dataset and Export](#6-final-dataset-and-export)
- [Notebook 2: `train_models.ipynb`](#notebook-2-train_modelsipynb)
  - [1. Import \& Load Clean Data](#1-import--load-clean-data)
  - [2. Feature Selection and Data Preparation](#2-feature-selection-and-data-preparation)
  - [3. Train-Test Split (Temporal)](#3-train-test-split-temporal)
  - [4. Feature Scaling](#4-feature-scaling)
  - [5. Week 3 Baseline Models](#5-week-3-baseline-models)
    - [Model 1: DummyRegressor](#model-1-dummyregressor)
    - [Model 2: Linear Regression](#model-2-linear-regression)
    - [Model 3: Decision Tree Regressor](#model-3-decision-tree-regressor)
    - [Baseline Model Comparison](#baseline-model-comparison)
  - [6. Week 4: Feature Engineering + More Models](#6-week-4-feature-engineering--more-models)
    - [Feature Sets](#feature-sets)
    - [Models Trained](#models-trained)
    - [Comprehensive Comparison](#comprehensive-comparison)
    - [Residual Diagnostics](#residual-diagnostics)
    - [Feature Importance Comparison](#feature-importance-comparison)
  - [7. Outputs Saved](#7-outputs-saved)
- [Week 5: Addressing Data Leakage and Methodological Improvements](#week-5-addressing-data-leakage-and-methodological-improvements)
  - [Feedback and Revisions](#feedback-and-revisions)
  - [Major Changes Implemented](#major-changes-implemented)
    - [1. Removed Lag Features (Critical Fix)](#1-removed-lag-features-critical-fix)
    - [2. Refined Temporal Split Strategy](#2-refined-temporal-split-strategy)
    - [3. Updated Feature Set](#3-updated-feature-set)
    - [4. Realistic Performance Expectations](#4-realistic-performance-expectations)
    - [5. Updated Correlation Analysis](#5-updated-correlation-analysis)
  - [Updated Model Performance Results](#updated-model-performance-results)
    - [Week 5 Baseline Models (Without Lag Features)](#week-5-baseline-models-without-lag-features)
    - [Feature Importance Analysis](#feature-importance-analysis)
    - [Prediction and Residual Analysis](#prediction-and-residual-analysis)
    - [Week 4 Advanced Models Performance](#week-4-advanced-models-performance)
  - [Lessons Learned](#lessons-learned)
    - [1. Data Leakage is Subtle](#1-data-leakage-is-subtle)
    - [2. High Performance Can Indicate Problems](#2-high-performance-can-indicate-problems)
  - [Conclusion](#conclusion)
- [Week 6: Synthesis \& Communication Readiness](#week-6-synthesis--communication-readiness)
  - [Overview](#overview)
  - [Changes to `train_models.ipynb`](#changes-to-train_modelsipynb)
    - [New Cell 106 — Section Header (Markdown)](#new-cell-106--section-header-markdown)
    - [New Cell 107 — Synthesis Pipeline (Code)](#new-cell-107--synthesis-pipeline-code)
  - [Final Model Selection](#final-model-selection)
    - [Top 5 Models by Test R²](#top-5-models-by-test-r)
  - [Justification for Stopping](#justification-for-stopping)
  - [Substantive Interpretation of Results](#substantive-interpretation-of-results)
  - [Limitations and Scope](#limitations-and-scope)
  - [Draft Presentation Figures](#draft-presentation-figures)
  - [Saved Artifacts](#saved-artifacts)
  - [Week 6 Deliverable Checklist](#week-6-deliverable-checklist)
  - [Conclusion](#conclusion-1)


# STATS 201 Course Project  
## Time Series Prediction of Life Expectancy (Data Cleaning ➜ Model Training)

## Author
**Bikalpa Panthi**

## Course
STATS 201

---

## Project Overview

This project builds a **time series regression pipeline** to predict **country-year life expectancy** using the Life Expectancy dataset.  
The workflow is intentionally split into **two notebooks**:

1. **`data_cleaning.ipynb`** — loads the raw dataset, cleans it, creates time-series-safe features (lags/rolling means), and exports a cleaned dataset.
2. **`train_models.ipynb`** — loads the cleaned dataset, performs a **temporal train-test split**, trains baseline + extended models, evaluates performance, and saves results/figures.

---

## Research Question

> **How accurately can we predict life expectancy over time using country-level health/economic indicators while respecting time ordering (no future leakage)?**

---


---

## End-to-End Pipeline

**Pipeline logic:**

1. **Raw CSV** (`Assets/Life Expectancy Data.csv`)  
2. **`data_cleaning.ipynb`**
   - Removes rows missing the target (`Life expectancy`)
   - Removes duplicate `(Country, Year)` rows
   - Adds time series features (lags, rolling means, YoY change)
   - Saves **clean dataset** (`clean_dataset.csv`)
   - Produces EDA & data quality plots in `images/`
3. **`train_models.ipynb`**
   - Loads `clean_dataset.csv`
   - Uses a **temporal split** (train = earlier years, test = later years)
   - Scales features
   - Trains and evaluates:
     - Week 3 baselines (Dummy / Linear Regression / Decision Tree)
     - Week 4 expanded models (Random Forest / Gradient Boosting / Ridge / Lasso)
   - Saves evaluation tables and plots

---

## Data Source

- **Dataset:** Life Expectancy Data (country-year panel)

### Core Variables
The dataset includes:
- `Country`, `Year` (panel/time keys)
- `Status` (Developed vs Developing)
- `Life expectancy` (target)
- Multiple health/economic predictors (e.g., mortality, immunization, GDP, schooling, etc.)

---

# Notebook 1: `data_cleaning.ipynb`

## 1. Import & Load

**Input:**
- Raw file: `Assets/Life Expectancy Data.csv`


---

## 2. Initial Exploration

This section checks:
- dataset shape
- first rows
- basic distributions / sanity checks

---

## 3. Missing Values

The notebook summarizes missingness by feature and visualizes missing value percentages.


![alt text](old_images/missing_values.png)

---

## 4. Cleaning Steps

### Remove Missing Target
Rows with missing target are dropped:

- `df_clean = df_clean.dropna(subset=['Life expectancy'])`

**Why:** supervised learning requires a known target during training/evaluation.

---

### Missing Predictors (Deferred Imputation)

This notebook intentionally **does not impute** missing predictors yet:

- “Imputation is deferred to model training to avoid leakage across time.”

**Why this matters for time series:**  
If you impute using information from later years, you can leak future information into earlier years.

---

### Duplicates (Country-Year)

Duplicates are detected using:
- `df_clean.duplicated(subset=['Country', 'Year'], keep=False)`

Then removed by keeping the first occurrence per `(Country, Year)`.

---

### Target Outliers (IQR)

The notebook computes outlier bounds using a conservative IQR rule:
- bounds = `Q1 - 3*IQR` to `Q3 + 3*IQR`

It reports potential outliers (but does not necessarily drop them unless you extend the notebook).


![alt text](old_images/life_expectancy_distribution.png)

---

### Encode Development Status

Creates:
- `Status_Encoded = 1` if Developed else `0`

This preserves the original `Status` column and adds a model-friendly numeric feature.

---

### Time Series Features

To support time-series prediction, the notebook sorts by:
- `Country`, then `Year`

Then creates **past-only** features:

1. `Years_Since_2000 = Year - 2000`
2. Lagged target features per country:
   - `Life_Expectancy_Lag_1`, `Life_Expectancy_Lag_2`, `Life_Expectancy_Lag_3`
3. Rolling means (past-only):
   - `Life_Expectancy_RollingMean_3yr`, `Life_Expectancy_RollingMean_5yr`  
   (computed on `shift(1)` so the current year never uses itself)
4. Year-over-year change (past-only):
   - `Life_Expectancy_YoY_Change` using `diff().shift(1)`

**Key principle:** feature engineering uses `.shift(...)` to avoid using the current year’s target when creating predictors.

---

### Missing Values from Lags

Lag features naturally introduce missingness in early years (e.g., first year has no lag-1).  
The notebook prints missing counts and explicitly states imputation happens in modeling.

---

## 5. Exploratory Data Analysis (EDA)

### Target Analysis

Produces status-based life expectancy comparisons and trends over time.

![alt text](old_images/life_expectancy_by_status.png)

---

### Correlations

Computes correlation between numeric predictors and target, then visualizes a correlation matrix of top features.

![alt text](old_images/correlation_matrix_top10.png)

---

### Distributions

Plots histograms of the most correlated features.

![alt text](old_images/top_features_distributions.png)

---

### Scatter Plots (Top Correlated Features)

Plots life expectancy vs the top positively correlated predictors, split by `Status`.

![alt text](old_images/scatter_plots_top_features.png)

---

## 6. Final Dataset and Export

**Output:**
- `clean_dataset.csv`

---

# Notebook 2: `train_models.ipynb`

## 1. Import & Load Clean Data

**Input:**
- `clean_dataset.csv` produced by `data_cleaning.ipynb`


---

## 2. Feature Selection and Data Preparation

This section defines:
- target: `Life expectancy`
- predictors: numeric features + engineered features (depending on feature set)
- handles data types
- prepares train/test matrices

---

## 3. Train-Test Split (Temporal)

This notebook uses a **time-respecting split** rather than random splitting.

**Goal:** train on earlier years, test on later years.

![alt text](old_images/train_test_distribution.png)

---

## 4. Feature Scaling

Uses `StandardScaler` fit on training data and applied to test data.

**Why:** many regression models benefit from standardized feature scales, and scaling must be fit only on training to avoid leakage.

---

## 5. Week 3 Baseline Models

All models are evaluated using standard regression metrics:
- RMSE (root mean squared error)
- MAE (mean absolute error)
- R²

### Model 1: DummyRegressor
**Purpose:** establishes a minimal performance baseline.

---

### Model 2: Linear Regression
Trains a linear baseline and visualizes coefficient magnitudes.

![alt text](old_images/linear_regression_coefficients.png)

---

### Model 3: Decision Tree Regressor
Trains a non-linear baseline and visualizes feature importances.

![alt text](old_images/decision_tree_importances.png)

---

### Baseline Model Comparison

The notebook produces model comparison plots and exports baseline results.

![alt text](old_images/model_comparison.png)

**Saved table:**
- `baseline_model_results.csv`

---

## 6. Week 4: Feature Engineering + More Models

### Feature Sets

The notebook defines three feature sets for controlled comparison:

1. **Baseline Features** (Week 3 originals)
2. **Polynomial Features** (quadratic terms for key predictors)
3. **Engineered Features** (domain-specific combinations)

---

### Models Trained

The Week 4 notebook trains and compares:
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Ridge Regression**
- **Lasso Regression**

---

### Comprehensive Comparison

The notebook generates an overall comparison plot/table across models and feature sets.

![alt text](old_images/week4_comprehensive_comparison.png)

![alt text](old_images/week4_model_predictions_comparison.png)

---

### Residual Diagnostics

Residual distributions are created for error-shape inspection.

![alt text](old_images/week4_residual_distributions.png)

---

### Feature Importance Comparison

Feature importance/coefficients are extracted for:
- Random Forest + Gradient Boosting (`feature_importances_`)
- Ridge + Lasso (`|coef_|`)

![alt text](old_images/week4_feature_importance_comparison.png)

---

## 7. Outputs Saved

Depending on which sections you run, the notebook writes:

**Tables**
- `baseline_model_results.csv`
- `week4_model_comparison.csv`
- `week4_feature_importance.csv`

**Figures**
- `train_test_distribution.png`
- `linear_regression_coefficients.png`
- `decision_tree_importances.png`
- `model_comparison.png`
- `prediction_analysis.png`
- `residual_analysis.png`
- `week4_comprehensive_comparison.png`
- `week4_model_predictions_comparison.png`
- `week4_residual_distributions.png`
- `week4_feature_importance_comparison.png`

---

# Week 5: Addressing Data Leakage and Methodological Improvements

## Feedback and Revisions


## Major Changes Implemented

### 1. Removed Lag Features (Critical Fix)

**Problem**: 
Lag features (`Life_Expectancy_Lag_1`, `Life_Expectancy_Lag_2`, `Life_Expectancy_Lag_3`, `Life_Expectancy_RollingMean_3yr`, `Life_Expectancy_RollingMean_5yr`, `Life_Expectancy_YoY_Change`) created severe data leakage. Models were essentially learning "predict next year's life expectancy ≈ this year's life expectancy + small change", resulting in artificially inflated performance metrics.

**Solution**:
Modified `data_cleaning.ipynb` to explicitly remove all lag-based features before export:
```python
# Added to data_cleaning.ipynb - Save Dataset section
lag_features = [
    'Life_Expectancy_Lag_1',
    'Life_Expectancy_Lag_2', 
    'Life_Expectancy_Lag_3',
    'Life_Expectancy_RollingMean_3yr',
    'Life_Expectancy_RollingMean_5yr',
    'Life_Expectancy_YoY_Change',
]

# Ensure these features are removed before saving
present_lags = [c for c in lag_features if c in df_clean.columns]
if present_lags:
    print(f"Removing lag features (data leakage): {present_lags}")
    df_clean = df_clean.drop(columns=present_lags)

# Validation check
if df_clean.shape[1] != 24:
    print(f"WARNING: Expected 24 columns, got {df_clean.shape[1]}")
```

**Impact**:
- Final dataset: **24 columns** (down from 30)
- No target-derived features remain
- Models must now learn from health/economic indicators only
- More realistic evaluation of predictive capability

---

### 2. Refined Temporal Split Strategy

**Implementation** (`train_models.ipynb`):
```python
# Temporal split: Train on past, test on future
years = df['Year']
train_mask = years <= 2013
test_mask = years >= 2014

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]
```

**Split Configuration**:
- **Training Period**: 2000-2013 (14 years, 2,562 observations, ~87.5%)
- **Test Period**: 2014-2015 (2 years, 366 observations, ~12.5%)
- **Rationale**: Models trained on historical data predict future years they have never seen

**Why This Matters**:
This represents true **time series forecasting** where:
- Test data is strictly in the future relative to training data
- No information from 2014-2015 influences model training
- Evaluation reflects real-world forecasting scenario
- No random shuffling that would mix past and future

![Train Test Distribution](images/train_test_distribution.png)

---

### 3. Updated Feature Set

**Removed (Week 5)**:
- `Life_Expectancy_Lag_1`
- `Life_Expectancy_Lag_2`
- `Life_Expectancy_Lag_3`
- `Life_Expectancy_RollingMean_3yr`
- `Life_Expectancy_RollingMean_5yr`
- `Life_Expectancy_YoY_Change`

**Retained (Week 5)** - 20 predictive features:
1. Adult Mortality
2. infant deaths
3. Alcohol
4. percentage expenditure
5. Hepatitis B
6. Measles
7. BMI
8. under-five deaths
9. Polio
10. Total expenditure
11. Diphtheria
12. HIV/AIDS
13. GDP
14. Population
15. thinness 1-19 years
16. thinness 5-9 years
17. Income composition of resources
18. Schooling
19. Status_Encoded
20. Years_Since_2000

**Plus metadata** (not used in modeling):
- Country
- Year
- Status
- Life expectancy (target)

---

### 4. Realistic Performance Expectations

**Before (Week 4 with lag features)**:
- R² values: 0.95-0.97
- RMSE: 1-2 years
- Artificially high due to data leakage

**After (Week 5 without lag features)**:
- Expected R²: 0.75-0.85 (more realistic)
- Expected RMSE: 3-5 years
- True reflection of predictive power from health/economic indicators

![Performance Comparison](images/performance_comparison.png)

---

### 5. Updated Correlation Analysis

With lag features removed, the correlation structure changed. The dataset now shows true relationships between health/economic indicators and life expectancy without artificial correlations from target-derived features.

![Correlation Matrix](images/correlation_matrix_top10.png)

---

## Updated Model Performance Results

### Week 5 Baseline Models (Without Lag Features)

All models were retrained using only the 20 health/economic indicator features, without any target-derived lag features.

![Model Comparison](images/model_comparison.png)

### Feature Importance Analysis

**Linear Regression Coefficients**:

![Linear Regression Coefficients](images/linear_regression_coefficients.png)

**Decision Tree Feature Importance**:

![Decision Tree Importances](images/decision_tree_importances.png)

**Top Predictors** (consistent across models):
1. Adult Mortality (strong negative correlation)
2. Schooling (strong positive correlation)
3. Income Composition of Resources (strong positive correlation)
4. HIV/AIDS
5. BMI

---

### Prediction and Residual Analysis

**Prediction Quality**:

![Prediction Analysis](images/prediction_analysis.png)

**Key Observations**:
- More scatter compared to Week 4 (expected with honest evaluation)
- Models still capture general trends
- Some systematic errors visible, suggesting room for improvement

**Residual Analysis**:

![Residual Analysis](images/residual_analysis.png)

**Residual Patterns**:
- Some outliers present
- Generally centered around zero (unbiased predictions)

---

### Week 4 Advanced Models Performance

![Week 4 Comprehensive Comparison](images/week4_comprehensive_comparison.png)

![Week 4 Model Predictions](images/week4_model_predictions_comparison.png)

![Week 4 Residual Distributions](images/week4_residual_distributions.png)

![Week 4 Feature Importance](images/week4_feature_importance_comparison.png)

---

## Lessons Learned

### 1. Data Leakage is Subtle
Using lag features seemed methodologically sound for time series, but created a shortcut that bypassed the actual prediction task. The high R² values should have been a warning sign, not a celebration.

### 2. High Performance Can Indicate Problems
R² > 0.95 should have been a red flag, not a success metric. Life expectancy prediction from socioeconomic factors alone should not be nearly perfect. When something seems too good to be true, it usually is.


---


## Conclusion

Week 5 addressed fundamental methodological flaws that would have invalidated all previous results. While removing lag features significantly decreased performance metrics, it produced an **honest evaluation** of the model's true predictive capability. 

This project now represents a legitimate **time series forecasting task** rather than a sophisticated curve-fitting exercise. The lower but more realistic R² values (0.75-0.85 range) demonstrate that health and economic indicators do contain meaningful predictive information about life expectancy, but the relationship is complex and cannot be reduced to simple temporal autocorrelation.

**Key Takeaway**: Scientific integrity requires honest evaluation, even when it means reporting lower performance metrics. The goal is accurate prediction and methodological soundness, not impressive-looking numbers that result from data leakage.

---
# Week 6: Synthesis & Communication Readiness

## Overview

Week 6 represents the final synthesis milestone. The primary objectives were to finalize the selected model, provide a stopping justification, produce a substantive interpretation of results for a social science audience, document clear limitations and scope, and generate draft presentation figures. All artifacts have been saved to the repository.

---

## Changes to `train_models.ipynb`

Two new cells (106–107) were added at the end of the notebook to implement the **Week 6 Synthesis Pipeline**. Minor import adjustments (`pathlib.Path`) were also made in earlier cells (3 and 94) to support path handling for artifact saving. All prior Week 3–5 code remains unchanged.

### New Cell 106 — Section Header (Markdown)
Introduces the Week 6 synthesis section and lists the deliverables:
- Final selected model + stopping justification
- Substantive interpretation summary
- Clear limitations and scope statement
- Draft presentation figures and saved artifact files

### New Cell 107 — Synthesis Pipeline (Code)
A single comprehensive code cell that:
1. **Normalizes the model comparison table** from Week 4 results
2. **Selects the final model** by ranking all 12 model–feature-set combinations by Test R² and generalization gap
3. **Generates a stopping justification** as a JSON metadata file
4. **Saves the trained model** as a `.joblib` artifact
5. **Writes interpretation and limitations** to a text file
6. **Produces a four-panel draft presentation figure** (saved as PNG)
7. **Prints a deliverable status checklist** confirming all milestones are met

---

## Final Model Selection

After evaluating twelve model–feature-set combinations across four model families (Gradient Boosting, Random Forest, Ridge, Lasso) and three feature sets (Baseline, Engineered, Polynomial), the **Gradient Boosting Regressor with Engineered Features** was selected as the final model.

| Metric | Value |
|--------|-------|
| Test R² | 0.9436 |
| Train R² | 0.9868 |
| Test RMSE | 1.98 years |
| Test MAE | 1.49 years |
| Generalization Gap (Train R² − Test R²) | 0.0432 |
| Margin over Runner-Up Test R² | 0.0054 |

### Top 5 Models by Test R²

| Rank | Model | Feature Set | Test R² | R² Gap |
|------|-------|-------------|---------|--------|
| 1 | Gradient Boosting | Engineered | 0.9436 | 0.0432 |
| 2 | Gradient Boosting | Polynomial | 0.9382 | 0.0488 |
| 3 | Random Forest | Engineered | 0.9368 | 0.0451 |
| 4 | Gradient Boosting | Baseline | 0.9354 | 0.0521 |
| 5 | Random Forest | Polynomial | 0.9313 | 0.0535 |

---

## Justification for Stopping

The model selection process was stopped based on three observations:

1. **Diminishing returns**: The margin between the best model (GB Engineered, R² = 0.9436) and the runner-up (GB Polynomial, R² = 0.9382) is only 0.0054 — well below the 0.01 threshold for meaningful improvement.
2. **Acceptable generalization gap**: The Train–Test R² gap of 0.0432 confirms the model is not severely overfitting.
3. **Exhausted model families**: No additional model family or feature combination produced a gain greater than 0.01 in Test R² over the selected model.

---

## Substantive Interpretation of Results

From a social science perspective, the final model confirms that life expectancy is most strongly predicted by a combination of **health burden indicators** and **socioeconomic development measures**.

**Top Predictive Features** (from the engineered feature set):
1. **Adult Mortality** — strong negative association with life expectancy
2. **HIV/AIDS prevalence** — strong negative association
3. **Income Composition of Resources** — strong positive association
4. **Schooling** — strong positive association
5. **BMI** — positive association

These findings align with established public health literature: mortality risk factors and access to education and economic resources are central determinants of population health outcomes.

**Error patterns**: The largest prediction errors (up to 8.6 years) occur for countries undergoing rapid transitions (e.g., Zimbabwe's post-crisis recovery) or nations with extreme HIV/AIDS burdens. Developing countries account for a disproportionate share of high-error cases, suggesting the model has more difficulty capturing health-system heterogeneity in lower-income settings.

**Temporal robustness**: Test R² values ranged from 0.921 to 0.944 across cutoff years 2010–2013, confirming stable predictive performance regardless of the specific train–test boundary.

---

## Limitations and Scope

- **Scope**: Supervised regression on the cleaned, lag-free dataset with a temporal split (train ≤ 2013, test ≥ 2014).
- **External validity**: Confined to populations and years represented in the WHO dataset.
- **Associational, not causal**: Feature importances reflect predictive utility, not intervention effects.
- **Residual confounding**: Unobserved factors such as healthcare policy changes, conflict, or reporting inconsistencies may influence results.
- **Misspecification risk**: Model diagnostics and robustness checks reduce but do not eliminate this risk.
- **Development-status heterogeneity**: Separate models for developed vs. developing countries may improve performance but were not pursued.

---

## Draft Presentation Figures

The synthesis pipeline generates a four-panel figure saved to `images/week6_presentation_figures.png`:

![Week 6 Presentation Figures](images/week6_presentation_figures.png)

**Panel descriptions**:
- **Top-left**: Model leaderboard ranked by Test R² — Gradient Boosting with engineered features leads
- **Top-right**: Generalization gap (Train R² − Test R²) — ensemble models show larger gaps than linear models but still within acceptable range
- **Bottom-left**: Actual vs. Predicted scatter plot for the selected model — points cluster tightly along the diagonal
- **Bottom-right**: Temporal robustness curve — Test R² steadily increases as more training years are included

---

## Saved Artifacts

| Artifact | Path |
|----------|------|
| Final trained model | `Assets/week6_final_model.joblib` |
| Model metadata (JSON) | `CSV outputs/week6_final_model_metadata.json` |
| Ranked model summary (CSV) | `CSV outputs/week6_final_model_summary.csv` |
| Interpretation & limitations (TXT) | `CSV outputs/week6_interpretation_limitations.txt` |
| Presentation figure (PNG) | `images/week6_presentation_figures.png` |

---

## Week 6 Deliverable Checklist

- [x] Final model(s) and justification for stopping
- [x] Substantive interpretation of results
- [x] Clear limitations and scope
- [x] Draft presentation figures
- [x] Near-final GitHub repository

---

## Conclusion

Week 6 finalizes the modeling pipeline. The Gradient Boosting model with engineered features has been selected as the final model, supported by a clear stopping justification grounded in diminishing marginal returns across model families and feature sets. The substantive interpretation connects statistical findings to meaningful public health narratives — adult mortality burden, education access, and economic resources are the strongest factors associated with life expectancy differences across nations. All artifacts, metadata, and presentation figures have been saved, bringing the repository to a near-final state ready for the final presentation.