"""Leakage-safe Phase 1 XGBoost predictions used as global context for Phase 2 U-Net."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .data import PHASE1_CATEGORICAL_FEATURES, PHASE1_NUMERIC_FEATURES


def make_phase1_xgb_pipeline():
    """Recreate the Phase 1 XGBoost feature pipeline and fixed model settings."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, PHASE1_NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, PHASE1_CATEGORICAL_FEATURES),
    ])
    model = XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def fit_leakage_safe_canopy_predictions(
    feature_engineered_path,
    split_manifest,
    results_dir,
):
    """Generate OOF training predictions and train-only predictions for val/test.

    The Phase 1 model never sees a training case's own scalar target when generating
    the global-context channel used to train the hybrid U-Net. Validation and test
    cases are predicted only by a model fitted on Phase 2 training families.
    """
    feature_engineered_path = Path(feature_engineered_path)
    results_dir = Path(results_dir)
    if not feature_engineered_path.exists():
        raise FileNotFoundError(
            "Phase 1 feature_engineered_dataset.csv is missing. Run the existing ML_01 "
            "and ML_03 notebooks first; Phase 2 does not recreate or overwrite Phase 1 outputs."
        )

    phase1 = pd.read_csv(feature_engineered_path)
    target_column = "target_U_canopy_mean_m_s"
    required_columns = (
        ["simulation_id", "family_id", target_column]
        + PHASE1_NUMERIC_FEATURES
        + PHASE1_CATEGORICAL_FEATURES
    )
    missing = [column for column in required_columns if column not in phase1.columns]
    if missing:
        raise KeyError(f"Phase 1 engineered table is missing required columns: {missing}")

    phase1 = phase1.drop_duplicates(subset=["simulation_id"]).copy()
    joined = phase1.merge(split_manifest, on=["simulation_id", "family_id"], how="inner", validate="one_to_one")
    if len(joined) != len(split_manifest):
        raise ValueError(
            "The Phase 1 engineered table and Phase 2 split manifest do not contain the same case IDs."
        )

    train = joined[joined["split"] == "train"].reset_index(drop=True)
    validation = joined[joined["split"] == "validation"].copy()
    test = joined[joined["split"] == "test"].copy()
    if train["family_id"].nunique() < 5:
        raise ValueError("At least five training families are required for grouped OOF predictions.")

    X_train = train[PHASE1_NUMERIC_FEATURES + PHASE1_CATEGORICAL_FEATURES]
    y_train = train[target_column].astype(float)
    groups = train["family_id"]
    oof_predictions = np.full(len(train), np.nan, dtype=np.float64)

    grouped_cv = GroupKFold(n_splits=5)
    for fold_number, (fit_indices, held_indices) in enumerate(
        grouped_cv.split(X_train, y_train, groups), start=1
    ):
        fit_families = set(groups.iloc[fit_indices])
        held_families = set(groups.iloc[held_indices])
        if fit_families & held_families:
            raise AssertionError("Family leakage occurred in the Phase 1 OOF hybrid stage.")
        fold_model = make_phase1_xgb_pipeline()
        fold_model.fit(X_train.iloc[fit_indices], y_train.iloc[fit_indices])
        oof_predictions[held_indices] = fold_model.predict(X_train.iloc[held_indices])
        print(
            f"Phase 1 XGBoost OOF fold {fold_number}: "
            f"fit families={len(fit_families)}, held-out families={len(held_families)}"
        )

    if np.isnan(oof_predictions).any():
        raise AssertionError("Some training cases did not receive an out-of-fold Phase 1 prediction.")

    final_phase1_model = make_phase1_xgb_pipeline()
    final_phase1_model.fit(X_train, y_train)

    prediction_rows = []
    for index, row in train.iterrows():
        prediction_rows.append({
            "simulation_id": row["simulation_id"],
            "split": "train",
            "family_id": row["family_id"],
            "actual_canopy_mean_m_s": float(row[target_column]),
            "predicted_canopy_mean_m_s": float(oof_predictions[index]),
            "prediction_method": "5-fold family-grouped out-of-fold",
        })

    for split_name, subset in [("validation", validation), ("test", test)]:
        if subset.empty:
            continue
        predictions = final_phase1_model.predict(
            subset[PHASE1_NUMERIC_FEATURES + PHASE1_CATEGORICAL_FEATURES]
        )
        for (_, row), prediction in zip(subset.iterrows(), predictions):
            prediction_rows.append({
                "simulation_id": row["simulation_id"],
                "split": split_name,
                "family_id": row["family_id"],
                "actual_canopy_mean_m_s": float(row[target_column]),
                "predicted_canopy_mean_m_s": float(prediction),
                "prediction_method": "XGBoost fitted on Phase 2 training families only",
            })

    prediction_table = pd.DataFrame(prediction_rows).sort_values("simulation_id").reset_index(drop=True)
    prediction_table.to_csv(results_dir / "hybrid_phase1_canopy_predictions.csv", index=False)
    joblib.dump(final_phase1_model, results_dir / "models" / "hybrid_phase1_xgboost.joblib")

    train_predictions = prediction_table.loc[
        prediction_table["split"] == "train", "predicted_canopy_mean_m_s"
    ].to_numpy(dtype=float)
    canopy_stats = {
        "mean": float(train_predictions.mean()),
        "std": float(train_predictions.std(ddof=0)) or 1.0,
        "fit_scope": "training cases only; training values are group-wise out-of-fold predictions",
    }
    with open(results_dir / "hybrid_canopy_channel_stats.json", "w", encoding="utf-8") as file:
        json.dump(canopy_stats, file, indent=2)

    test_rows = prediction_table[prediction_table["split"] == "test"]
    scalar_metrics = {
        "test_cases": int(len(test_rows)),
        "phase1_canopy_MAE_m_s": float(mean_absolute_error(
            test_rows["actual_canopy_mean_m_s"], test_rows["predicted_canopy_mean_m_s"]
        )),
        "phase1_canopy_RMSE_m_s": float(np.sqrt(mean_squared_error(
            test_rows["actual_canopy_mean_m_s"], test_rows["predicted_canopy_mean_m_s"]
        ))),
        "phase1_canopy_R2": float(r2_score(
            test_rows["actual_canopy_mean_m_s"], test_rows["predicted_canopy_mean_m_s"]
        )),
    }
    with open(results_dir / "hybrid_phase1_scalar_test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(scalar_metrics, file, indent=2)

    prediction_lookup = dict(zip(
        prediction_table["simulation_id"],
        prediction_table["predicted_canopy_mean_m_s"],
    ))
    return prediction_lookup, canopy_stats, prediction_table, scalar_metrics
