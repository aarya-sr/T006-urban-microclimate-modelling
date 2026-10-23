# Phase 1 methodology and viva notes

## Problem and target
The Phase 1 AML experiment predicts one scalar: canopy-averaged wind speed (m/s) for an idealized layout and sampled wind direction. It does not predict temperature or a spatial field.

## Why family-held-out validation?
Each layout contributes related direction-specific rows, and layouts are grouped into plan families. A random row split could place near-related cases in both train and test sets, making generalization appear easier. GroupKFold on `family_id` holds out entire families. Layout-held-out results are secondary because related families can still appear across folds.

## Why these models?
- Mean baseline: verifies that predictors improve on a constant reference.
- Ridge: linear reference with L2 shrinkage; standardization matters because the penalty depends on coefficient scale.
- Random Forest: bagged trees can model thresholds and interactions.
- XGBoost: boosted trees learn sequentially from residual errors.

The tree models are Phase 1 tabular AML baselines, not neural networks.

## Feature rationale
- Plan-area density: compact measure of built-up fraction.
- Squared density: tests a simple nonlinear density response.
- Height spread/range and coefficient of variation: summarize vertical geometry. Mean height is constant at 16 m in this subset and is not used as a predictor.
- Wind-direction sine/cosine: circular encoding.
- Configuration, horizontal configuration and density class: categorical geometry descriptors.
- IDs and family labels are excluded from prediction; family/layout identifiers are used for grouping only.

## Metrics
- MAE: average absolute error in m/s.
- RMSE: square root of mean squared error; larger misses count more.
- Normalized RMSE: RMSE divided by the observed target range; the denominator must be stated.
- R²: relative to a constant-mean reference and may be negative.

## Limitations to state clearly
1. Only 224 idealized cases, 112 layouts and 16 families are in the current scalar experiment.
2. Only two wind directions are represented in this subset.
3. ZIP archives provide sampled field/topology/profile members, not complete spatial fields for every case.
4. Layout summaries cannot encode local wake and street-canyon patterns.
5. A full raster-to-field CNN/U-Net and field MAE/normalized RMSE/error maps are Phase 2 deliverables, not current Phase 1 claims.
6. Results from 16 families have uncertainty; bootstrap intervals are approximate.

## Viva prompts each team member should be able to answer
1. How does ML_01 join target and geometry metadata, and how is the join validated?
2. What exactly does the target represent and what does it not represent?
3. Why is a random split inappropriate here?
4. What does Ridge's alpha do? Why scale numeric features?
5. What is the difference between Random Forest bagging and XGBoost boosting?
6. Why can feature importance not establish causality?
7. What does negative R² mean?
8. What would need to change to train a U-Net on spatial fields?
