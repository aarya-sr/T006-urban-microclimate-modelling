# Five-member Phase 1 work split

| Member | Assigned step(s) | Deliverable / viva evidence |
|---|---|---|
| Suhaani | 01 Dataset understanding and data dictionary; 05 Ridge Regression baseline | Dataset provenance, data dictionary updates, linear baseline and preprocessing explanation |
| Aarya | 02 EDA and target definition; 07 Evaluation and comparison | EDA notebook, target rationale, metric table and result interpretation |
| Palak | 03 Leakage audit and feature engineering | Predictor whitelist, target exclusion audit, metadata-to-feature logic |
| Jashvitha | 04 Family-held-out split and preprocessing | Family group assignment proof, leakage demonstration, fold sizes and train-fold-only preprocessing |
| Samarth | 06 Random Forest and XGBoost | Model configuration, predictions, tree-model discussion and limitations |


## Shared literature review and report

All five members should read ML_00, verify the suggested pre-read metadata against the faculty-provided source, and contribute to the final LaTeX/IEEE report. Notebook ownership supports coordination; it does not replace each member’s individual understanding for viva.

## Validation ownership

The primary protocol is 5-fold `GroupKFold` over the 16 `family_id` groups (footprint + plan-area density); leave-one-family-out and a layout-grouped interpolation split are reported as secondary protocols. All 14 cases of a family (7 height patterns x 2 wind directions) go to one and only one fold. The grouped-split and model-evaluation notebooks assert train/test group disjointness for the relevant folds and protocols.
