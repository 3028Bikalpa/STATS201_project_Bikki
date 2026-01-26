# Predicting Population Decline Using Demographic and Migration Data
## Project Overview
- [Predicting Population Decline Using Demographic and Migration Data](#predicting-population-decline-using-demographic-and-migration-data)
  - [Project Overview](#project-overview)
  - [Research Question](#research-question)
  - [Project Description](#project-description)
  - [Unit of Analysis](#unit-of-analysis)
  - [Data Source](#data-source)
  - [Data Preparation and Feature Engineering](#data-preparation-and-feature-engineering)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Train/Test Split Strategy](#traintest-split-strategy)
  - [Models Implemented](#models-implemented)
    - [Dummy Classifier](#dummy-classifier)
    - [Logistic Regression](#logistic-regression)
    - [Decision Tree Classifier](#decision-tree-classifier)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Results Summary](#results-summary)
    - [Dummy Classifier](#dummy-classifier-1)
    - [Logistic Regression](#logistic-regression-1)
    - [Decision Tree Classifier (Depth = 5)](#decision-tree-classifier-depth--5)
  - [Interpretation](#interpretation)
  - [Future Work](#future-work)

This project investigates whether demographic and migration indicators can provide reliable early warning signals of population decline. Using historical population and migration data, the goal is to build predictive models that determine whether a country will experience population decline within the next five years.

The problem is formulated as a **binary classification task**, where models are trained on historical data and evaluated under realistic, time-aware forecasting conditions.

---

## Research Question

**What demographic and migration indicators provide reliable early warning signals of population decline, and what does this reveal about the mechanisms driving demographic transitions?**

---

## Project Description

This is a **time series machine learning project** based on a global dataset covering **186 countries** from **1960 to 2023**. The dataset includes population size, migration statistics, and related demographic indicators.

- Data up to **2010** is used as the training set.
- Data from **2011 to 2023** is used as the test set.

The target variable is `future_decline`, defined as:

- `1` → Population declines within the next five years  
- `0` → Population does not decline within the next five years  

---

## Unit of Analysis

The unit of analysis is a **country–year observation**, where each row corresponds to one country in one year.

---

## Data Source

The dataset used in this project was obtained from Kaggle:

https://www.kaggle.com/datasets/hashimkhanwazir/global-population-and-migration-dataset

- **Countries:** 186  
- **Time span:** 64 years (1960–2023)  
- **Total observations:** 11,904  

---

## Data Preparation and Feature Engineering

The dataset was cleaned and prepared through the following steps:

- Standardized column names and converted variables to appropriate data types.
- Removed observations with incomplete future information to prevent data leakage.
- Created a forward-looking target variable using a five-year prediction window.
- Engineered meaningful features, including:
  - Lagged population and migration values
  - Population growth rates
  - Rolling averages
  - Change variables
  - Domain-specific indicators such as migration intensity

---

## Exploratory Data Analysis

Exploratory analysis was conducted to better understand trends and patterns in the data, including:

- Population and migration trends over time
- Class imbalance between decline and non-decline cases
- Temporal patterns across countries
- Feature distributions and correlations

![Class distribution in train and test set](Assets/image.png)  
*Class distribution in training and test sets*

---

## Train/Test Split Strategy

A **temporally ordered split** was used to reflect real-world forecasting conditions:

- Training data: years up to 2010  
- Test data: years from 2011 to 2023  

This approach ensures the model learns from historical trends and is evaluated on future, unseen data.

---

## Models Implemented

Three baseline models were implemented and compared:

### Dummy Classifier

A baseline model that predicts outcomes based on class frequency in the training data. It serves as a minimum benchmark for evaluating model performance.

### Logistic Regression

Logistic Regression was selected due to its interpretability and effectiveness in problems with linear relationships. It predicts population decline based on demographic and migration features.

### Decision Tree Classifier

The Decision Tree model captures non-linear relationships and interaction effects between features without requiring explicit transformations. This model demonstrated the strongest overall performance.

![Baseline Model Perfomance Comparision](Assets/image-1.png)  
*Comparison of Dummy Classifier, Logistic Regression, and Decision Tree Classifier*

---

## Evaluation Metrics

Model performance was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

Due to class imbalance and the importance of identifying decline events, particular emphasis was placed on **Recall** and **ROC-AUC**.

---

## Results Summary

### Dummy Classifier

- Accuracy: 0.8112  
- Precision: 0.0000  
- Recall: 0.0000  
- F1-score: 0.0000  

### Logistic Regression

- Accuracy: 0.8542  
- Precision: 0.6260  
- Recall: 0.5658  
- F1-score: 0.5944  
- ROC-AUC: 0.8416  

### Decision Tree Classifier (Depth = 5)
![Decision treee](Assets/decision_tree.png)

- Accuracy: 0.9227  
- Precision: 0.8807  
- Recall: 0.6833  
- F1-score: 0.7695  
- ROC-AUC: 0.9345  

---

## Interpretation

The results indicate that the **Decision Tree Classifier** outperformed both the Dummy Classifier and Logistic Regression model across all evaluation metrics. This suggests that population decline is driven by complex, non-linear interactions between demographic and migration variables.

While the strong performance of the Decision Tree is encouraging, it also raises the possibility of **overfitting**, highlighting the need for further validation and model refinement.

---

## Future Work

Potential extensions of this project include:

- Addressing class imbalance using resampling or cost-sensitive learning
- Exploring ensemble methods such as Random Forests or Gradient Boosting
- Hyperparameter tuning and cross-validation
- Model interpretability analysis (e.g., SHAP values)
- Policy-oriented analysis of early warning indicators

---
