# STATS 201 Course Project
## Predicting National Life Expectancy with Socioeconomic and Health-System Indicators

Author: **Bikalpa Panthi**  
Course: **STATS 201**

---

## Important Scope Note
The folder `older_version/` is archival and **not** part of the active submission workflow.  
This README documents only the current project version using:
- `Assets/`
- `Codes/`
- `Images/`

---

## Research Question
**Which socioeconomic and health-system indicators most strongly predict national life expectancy, and how do different feature representations and modeling strategies affect predictive performance?**

---

## Project Overview
This project builds a full country-year prediction pipeline from raw WHO/World Bank files through model selection and final interpretation.

The active workflow is split across five notebooks:
1. `Codes/export.ipynb`
2. `Codes/data_cleaning.ipynb`
3. `Codes/exploratory_analysis.ipynb`
4. `Codes/train_models.ipynb`
5. `Codes/project_final.ipynb`

The final modeling setup uses a strict temporal split:
- **Train:** 2000-2017
- **Test:** 2018-2021

This ensures future years are never used to fit models.

---

## Repository Structure

```text
STATS201/
  Assets/
    full_data/                 # 14 source files (WHO/World Bank)
    cleaner_exports/           # merged panel, cleaned data, split files, model outputs
  Codes/
    export.ipynb
    data_cleaning.ipynb
    exploratory_analysis.ipynb
    train_models.ipynb
    project_final.ipynb
  Images/
    exploratory_*.png
    *_temporal.png
    project_final_*.png
```

---

## Data Inputs (`Assets/full_data`)
The active dataset is assembled from 14 files:
- `Adult_mortality.xlsx`
- `Diptheria.xlsx`
- `GDP_and_population.csv`
- `HIV.xlsx`
- `HepB3.xlsx`
- `Infant_deaths.xlsx`
- `Life_expectancy.xlsx`
- `Polio.xlsx`
- `Thinness.xlsx`
- `U5_mortality.xlsx`
- `alcohol_consumption.xlsx`
- `health_expenditure.xlsx`
- `overweight_adults.xlsx`
- `underweight_adults.xlsx`

---

## End-to-End Pipeline

### 1) Export and merge (`Codes/export.ipynb`)
- Reads all 14 raw files.
- Harmonizes country names and ISO3.
- Produces merged panel:
  - `Assets/cleaner_exports/health_panel_2000_2021.csv`

### 2) Cleaning and ML-ready tables (`Codes/data_cleaning.ipynb`)
- Applies feature/country completeness filters.
- Performs imputation and conservative clipping.
- Produces:
  - `health_panel_ml_clean.csv`
  - `health_panel_ml_numeric.csv`
  - dropped-feature/country summaries.

### 3) Exploratory analysis + split (`Codes/exploratory_analysis.ipynb`)
- Dataset profiling and feasibility checks.
- Correlation and trend analysis.
- Temporal split exports:
  - `train_temporal_2000_2017.csv`
  - `test_temporal_2018_2021.csv`
  - `temporal_split_summary.csv`

### 4) Modeling and diagnostics (`Codes/train_models.ipynb`)
- Baseline and advanced models.
- Representation comparison: `baseline`, `log_enhanced`, `polynomial`.
- Controlled robustness and residual diagnostics.
- Final model artifact + stopping justification.

### 5) Final synthesis (`Codes/project_final.ipynb`)
- Consolidates best model and representation.
- Answers the research question directly.
- Produces final summary table and synthesis figures.

---

## Key Exploratory Figures

### Life expectancy distribution
![Exploratory Distribution](Images/exploratory_life_expectancy_distribution.png)

### Mean life expectancy over time
![Exploratory Trend](Images/exploratory_life_expectancy_trend.png)

### Correlation structure
![Exploratory Correlation Heatmap](Images/exploratory_correlation_heatmap.png)

### Top predictor relationships
![Top Predictors](Images/exploratory_top_predictors_vs_target.png)

These plots show strong signal from mortality-related indicators and confirm enough temporal/cross-country variation to support predictive modeling.

---

## Modeling Strategy and Comparison

Models trained in `train_models.ipynb`:
- DummyRegressor
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Ridge
- Lasso

Each model is evaluated across feature representations:
- `baseline`
- `log_enhanced`
- `polynomial`

Evaluation metrics:
- Test R²
- Test RMSE
- Test MAE
- R² Gap (Train R² - Test R²)

### Model comparison figure
![Model Comparison Temporal](Images/model_comparison_temporal.png)

### Feature importance (best model)
![Feature Importance Temporal](Images/feature_importance_temporal.png)

### Prediction quality (test set)
![Prediction Analysis Temporal](Images/prediction_analysis_temporal.png)

---

## Robustness and Diagnostics

Controlled checks include:
- Temporal sensitivity across cutoff years.
- Feature group ablation.
- Sample-size sensitivity.
- Year-level stability.

### Robustness summary
![Robustness Summary](Images/robustness_summary_temporal.png)

### Residual diagnostics
![Residual Diagnostics](Images/residual_diagnostics_temporal.png)

Diagnostics outputs are saved in:
- `Assets/cleaner_exports/robustness_checks_temporal.csv`
- `Assets/cleaner_exports/residual_diagnostics_summary_temporal.csv`
- `Assets/cleaner_exports/influential_observations_temporal.csv`
- `Assets/cleaner_exports/error_analysis_detailed_temporal.csv`

---

## Final Model Selection
From `Assets/cleaner_exports/model_comparison_temporal.csv` and `project_final.ipynb`:

- **Best model:** `Random Forest (n_estimators=300, max_depth=15)`
- **Best feature representation:** `log_enhanced`
- **Best combination:** `Random Forest + log_enhanced`

Performance on temporal test set (2018-2021):
- **Test R²:** 0.988479
- **Test RMSE:** 0.754141
- **Test MAE:** 0.521731
- **R² Gap:** 0.011215

Stopping decision (from `final_model_justification_temporal.txt`):
- **Decision:** STOP and freeze final model.
- **Reason:** runner-up gain is negligible and robustness/generalization thresholds are satisfied.

---

## Final Synthesis Figures

### Indicator-family importance summary
![Final Indicator Importance Summary](Images/project_final_indicator_importance_summary.png)

### Top model-representation combinations
![Final Top Combinations](Images/project_final_top_combinations.png)

### Temporal sensitivity for final conclusion
![Final Temporal Sensitivity](Images/project_final_temporal_sensitivity.png)

These final figures summarize the answer to the research question:
- Mortality-related indicators dominate predictive power.
- Socioeconomic and risk-profile indicators add secondary signal.
- The strongest generalization comes from a non-linear ensemble with log-transformed economic/mortality magnitude features.

---

## Main Output Files

### Core datasets and artifacts (`Assets/cleaner_exports`)
- `health_panel_2000_2021.csv`
- `health_panel_ml_clean.csv`
- `health_panel_ml_numeric.csv`
- `train_temporal_2000_2017.csv`
- `test_temporal_2018_2021.csv`
- `temporal_split_summary.csv`
- `baseline_model_results_temporal.csv`
- `model_comparison_temporal.csv`
- `feature_importance_analysis_temporal.csv`
- `updated_results_summary_temporal.csv`
- `representation_wins_temporal.csv`
- `robustness_checks_temporal.csv`
- `residual_diagnostics_summary_temporal.csv`
- `influential_observations_temporal.csv`
- `error_analysis_detailed_temporal.csv`
- `error_by_year_temporal.csv`
- `error_by_target_quantile_temporal.csv`
- `final_model_justification_temporal.txt`
- `final_model_temporal.joblib`
- `project_final_summary.csv`

### Figures (`Images`)
- `exploratory_life_expectancy_distribution.png`
- `exploratory_life_expectancy_trend.png`
- `exploratory_correlation_heatmap.png`
- `exploratory_top_predictors_vs_target.png`
- `model_comparison_temporal.png`
- `feature_importance_temporal.png`
- `prediction_analysis_temporal.png`
- `robustness_summary_temporal.png`
- `residual_diagnostics_temporal.png`
- `project_final_indicator_importance_summary.png`
- `project_final_top_combinations.png`
- `project_final_temporal_sensitivity.png`

---

## Reproducibility: Execution Order
Run notebooks in this order:
1. `Codes/export.ipynb`
2. `Codes/data_cleaning.ipynb`
3. `Codes/exploratory_analysis.ipynb`
4. `Codes/train_models.ipynb`
5. `Codes/project_final.ipynb`

---

## Limitations and Scope
- Predictive modeling only; not causal inference.
- Country-level annual aggregates cannot capture within-country heterogeneity.
- Evaluation uses one temporal holdout window (2018-2021).
- Reporting quality and measurement consistency may differ by country-year.
- Best model is best for this dataset/split protocol, not necessarily universal.
