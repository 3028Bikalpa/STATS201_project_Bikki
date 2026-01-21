# Predicting Population Decline from Migration and Demographic Data

## Research Question

**Can past migration trends and demographic indicators be used to predict whether a country will enter a period of population decline within the next _N_ years?**

This project uses a global country–year dataset containing population and migration data for **186 countries** from **1960 to 2023**. The project is framed as a **supervised time-series prediction task**, where machine learning models are trained on historical observations and evaluated on future data using a temporal train–test split.

The goal of the project is to assess whether migration and demographic variables contain predictive information about future population decline and to interpret model behavior in a social science context rather than to make causal claims.

---

## Unit of Analysis

The unit of analysis for this project is a **country–year observation**, where each row corresponds to one country in one year.

---

## Data Source

This project uses a publicly available dataset from **Kaggle**:

- Dataset: *Global Population and Migration Dataset*
- Link: https://www.kaggle.com/datasets/hashimkhanwazir/global-population-and-migration-dataset

The dataset includes population and migration indicators for 186 countries over a 64-year period, resulting in **11,904 country–year observations**.

---

## Feasibility Assessment

The dataset provides sufficient temporal coverage and cross-national variation to support a supervised machine learning task. Key population and migration variables are observed prior to the prediction target, allowing for a clean temporal train–test split and avoiding information leakage.

While some heterogeneity across countries and missing values may be present, the overall size and structure of the dataset make it well-suited for training and evaluating predictive models of population decline.
