# Beef Eating Quality Prediction — ML Pipeline

An end-to-end machine learning project predicting beef eating quality 
scores from carcass grading data and post-mortem chill curve time-series, 
developed for a major Irish beef processor.

## Problem

Beef eating quality is highly variable and difficult to predict at the 
point of slaughter. The industry standard (MSA — Meat Standards Australia) 
uses fixed population-level coefficients that don't account for 
supply-chain-specific factors. The goal was to build a model calibrated 
to Irish cattle, grading conditions, and sensory panels that could 
outperform or complement the existing MSA framework.

## Approach

**Modelling**
- CatBoost regression selected for its native handling of mixed feature 
  types (categorical, numeric, ordinal, binary) without one-hot encoding
- Cut-specific models built for Loin, Topside, and other primals
- Achieved R² of ~0.68 on held-out data

**Feature Engineering**
- Novel membrane volume method applied to pH/temperature chill curve 
  time-series data to engineer meaningful post-mortem features
- pH × time × temperature interaction surface captured as area-under-curve 
  features, adding 7–11% incremental predictive importance over baseline

**Validation**
- GroupKFold cross-validation used with animal as the grouping key, 
  preventing data leakage across cuts from the same carcass
- ICC (Intraclass Correlation Coefficient) used to detect and correct 
  for panellist bias in sensory scores

**Target Variable**
- MQ4 eating quality score — a weighted composite of Tenderness (×0.3), 
  Juiciness (×0.1), Flavour (×0.3), and Overall Liking (×0.3)

## Key Findings

- Chill curve-derived features provide a novel predictive signal not 
  captured by the existing MSA grading framework
- Topside specification showed highest sensitivity to chill environment, 
  making it the most commercially actionable cut for intervention
- Panellist bias correction via ICC meaningfully improved model reliability

## Tech Stack

- Python (CatBoost, Pandas, Scikit-learn, Matplotlib)
- GroupKFold cross-validation
- ICC bias correction
- Time-series feature engineering
- Statistical sensory analysis

## Note

Data is proprietary and not included in this repository. A synthetic 
data notebook demonstrating the methodology is available in 
`synthetic_demo.ipynb`.
