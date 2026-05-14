# Beef Eating Quality Prediction — ML Pipeline

An end-to-end machine learning project predicting beef eating quality 
scores from carcass grading data and post-mortem chill curve time-series, 
developed for a large European food processor.

## Problem

Beef eating quality is highly variable and difficult to predict at the 
point of slaughter. The industry standard (MSA — Meat Standards Australia) 
uses fixed population-level coefficients that don't account for 
supply-chain-specific factors. The goal was to build a model calibrated 
to specific cattle supply chains, grading conditions, and sensory panels.

## Dataset Architecture

A key early decision was to move away from averaged panel scores 
(one mean score per animal per cut) and instead treat each individual 
panellist evaluation as a separate row.

This transformation expanded the dataset from approximately 400 
carcass-level observations to over 10,900 individual panellist 
evaluation records across four primal cuts — exposing the full 
within-animal disagreement that averaging would have hidden.

The dataset was constructed by joining two sources:
- **Carcass-level data** — physical grading measurements, chill curve 
  time-series, breed, kill date, fat cover, conformation
- **Panellist-level data** — individual sensory scores per cut per 
  assessor (Tenderness, Juiciness, Flavour, Overall Liking)

Joined on sample ID, with GroupKFold applied at animal level to 
prevent leakage across cuts from the same carcass.

## Approach

**Modelling**
- CatBoost regression selected for its native handling of mixed feature 
  types (categorical, numeric, ordinal, binary) without one-hot encoding
- Cut-specific models built for individual primals
- Benchmarked against the MSA industry standard framework

**Feature Engineering**
- Novel membrane volume method applied to pH/temperature chill curve 
  time-series data to engineer meaningful post-mortem features
- pH × time × temperature interaction surface captured as 
  area-under-curve features, providing incremental predictive signal 
  over baseline grading data

**Panellist Bias Correction**
- ANOVA confirmed panellist identity as a statistically significant 
  driver of score variance across Tenderness, Juiciness, and Flavour
- ICC (Intraclass Correlation Coefficient) used to quantify 
  measurement noise versus genuine between-animal quality signal
- Bias correction applied per panellist prior to modelling, 
  improving label reliability and raising the theoretical R² ceiling

**Validation**
- GroupKFold cross-validation with animal as the grouping key
- ICC used as a measurement quality metric to establish the 
  theoretical upper bound on model performance

**Target Variable**
- MQ4 eating quality score — a weighted composite of Tenderness, 
  Juiciness, Flavour, and Overall Liking

## MLflow Experiment Tracking

MLflow was used throughout the project lifecycle for full experiment 
reproducibility and model governance:

- **Experiment Tracking** — all model runs logged with hyperparameters, 
  training metrics, and cross-validation scores, enabling systematic 
  comparison across iterations and cut-specific model variants
- **Artefact Logging** — feature importance plots, residual 
  diagnostics, ICC reports, and validation outputs stored per run 
  for full traceability
- **Model Registry** — trained cut-specific models versioned and 
  registered, with staging and production tags applied as models 
  were validated and approved for client delivery

This setup ensured that every result presented to the client could 
be traced back to an exact model version, dataset snapshot, and 
parameter configuration.

## Tech Stack

- Python (CatBoost, Pandas, Scikit-learn, Matplotlib)
- MLflow (experiment tracking, model registry, artefact store)
- GroupKFold cross-validation
- ICC bias correction
- Time-series feature engineering
- Statistical sensory analysis

## Note

Data is proprietary and not included in this repository. A synthetic 
data notebook demonstrating the methodology is available in 
`synthetic_demo.ipynb`.
