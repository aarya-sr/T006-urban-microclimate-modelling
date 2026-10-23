# Five-member Phase 1 work split

| Member | Assigned step(s) | Deliverable / viva evidence |
|---|---|---|
| Suhaani | 01 Dataset understanding and data dictionary; 05 Ridge Regression baseline | Dataset provenance, data dictionary updates, linear baseline and preprocessing explanation |
| Aarya | 02 EDA and target definition; 07 Evaluation and comparison | EDA notebook, target rationale, metric table and result interpretation |
| Palak | 03 Leakage audit and feature engineering | Predictor whitelist, target exclusion audit, metadata-to-feature logic |
| Jashvitha | 04 Family-held-out split and preprocessing | Family group assignment proof, leakage demonstration, fold sizes and train-fold-only preprocessing |
| Samarth | 06 Random Forest and XGBoost | Model configuration, predictions, tree-model discussion and limitations |

All five should review the complete pipeline and report so each can explain the problem, data, model assumptions, grouped validation, results, and limitations in the individual viva. Make attributable commits or pull requests in the team's GitHub repository. Do not use `simulation_id`, `layout_id`, `family_id` or `configuration_metadata_label` as predictors.

## Validation ownership

The primary protocol is 5-fold `GroupKFold` over the 16 `family_id` groups (footprint + plan-area density); leave-one-family-out and a layout-grouped interpolation split are reported as secondary protocols. All 14 cases of a family (7 height patterns x 2 wind directions) go to one and only one fold. `src/train_models.py` asserts train/test group disjointness for every fold and protocol.
