"""Phase 1 XGBoost -> global context channel for the hybrid U-Net (replacement for dl_src/hybrid.py).

WHY THIS FILE IS SEPARATE FROM THE TORCH CODE
  * It never imports torch (the old version did, indirectly, through `from .data import ...`).
    On macOS, torch and xgboost each ship their own OpenMP runtime; loading both in one
    process is a well-known cause of "kernel crashed" with no Python traceback.
  * It can be run on its own:   python -m dl_src.hybrid      (from the repo root)

WHAT CHANGED IN THE LOGIC (see DL_05 for the numbers)
  1. Channel normalisation uses the mean/std of the TRUE training canopy target, not of the
     out-of-fold predictions. OOF predictions are shrunk (std 0.43 vs 0.83 for the real target),
     so the old z-scores reached +5.7 on val/test while training only ever saw -1.6..+2.5.
  2. Validation/test channel = average of the 5 fold models (each trained on 80% of the training
     families), instead of one model refit on all training families. This makes the channel at
     test time look statistically like the OOF channel the U-Net was trained on.
"""
import os

# Must be set before numpy / xgboost are imported. One thread is plenty for 224 rows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor  # keep this import ahead of anything that could load torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Identical to ML_03 / ML_06 (and to dl_src.data). DL_05 asserts this at run time.
PHASE1_NUMERIC_FEATURES = [
    "plan_area_density", "plan_area_density_sq", "height_std_m", "height_max_m",
    "height_min_m", "height_range_m", "height_skew", "frac_taller_than_mean",
    "wind_direction_sin", "wind_direction_cos", "height_coefficient_variation",
]
PHASE1_CATEGORICAL_FEATURES = ["configuration", "horizontal_configuration", "density_class"]
TARGET = "target_U_canopy_mean_m_s"

# Identical to XGB_PARAMS in ML_06 except n_jobs (4 -> 1, avoids OpenMP thread clashes).
XGB_PARAMS = dict(
    n_estimators=200, max_depth=2, learning_rate=0.03, min_child_weight=3,
    subsample=0.85, colsample_bytree=0.9, reg_lambda=2.0,
    objective="reg:squarederror", tree_method="hist", n_jobs=1, random_state=42,
)


def make_phase1_xgb_pipeline():
    """Same preprocessing + hyper-parameters as the Phase 1 XGBoost model."""
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), PHASE1_NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), PHASE1_CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", XGBRegressor(**XGB_PARAMS))])


def build_hybrid_channel(feature_engineered_path, split_manifest_path, results_dir, n_splits=5):
    """Create the leakage-safe Phase 1 prediction for every one of the 224 cases.

    train       -> out-of-fold prediction (family-grouped GroupKFold): the model that predicts a
                   case never saw that case's family.
    val / test  -> mean of the n_splits fold models, all fitted on training families only.
    Test and validation targets are never used to fit anything.
    """
    results_dir = Path(results_dir)
    (results_dir / "models").mkdir(parents=True, exist_ok=True)

    phase1 = pd.read_csv(feature_engineered_path).drop_duplicates(subset=["simulation_id"])
    split = pd.read_csv(split_manifest_path)[["simulation_id", "family_id", "split"]]
    joined = phase1.merge(split, on="simulation_id", how="inner", suffixes=("", "_dl"))
    if len(joined) != len(split):
        raise ValueError("Phase 1 table and Phase 2 split manifest do not contain the same cases.")
    if not (joined["family_id"] == joined["family_id_dl"]).all():
        raise ValueError("family_id differs between the Phase 1 table and the Phase 2 split.")

    features = PHASE1_NUMERIC_FEATURES + PHASE1_CATEGORICAL_FEATURES
    train = joined[joined["split"] == "train"].reset_index(drop=True)
    X_train, y_train, groups = train[features], train[TARGET].astype(float), train["family_id"]
    if groups.nunique() < n_splits:
        raise ValueError("Need at least as many training families as folds.")

    oof = np.full(len(train), np.nan)
    fold_models = []
    for fold, (fit_idx, held_idx) in enumerate(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups), 1):
        assert set(groups.iloc[fit_idx]).isdisjoint(set(groups.iloc[held_idx])), "family leakage"
        model = make_phase1_xgb_pipeline().fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof[held_idx] = model.predict(X_train.iloc[held_idx])
        fold_models.append(model)
        print(f"fold {fold}: fit on {groups.iloc[fit_idx].nunique()} families, "
              f"predicted {groups.iloc[held_idx].nunique()} unseen families")
    assert not np.isnan(oof).any()

    rows = [dict(simulation_id=r.simulation_id, split="train", family_id=r.family_id,
                 actual_canopy_mean_m_s=float(getattr(r, TARGET)), predicted_canopy_mean_m_s=float(p),
                 prediction_method=f"{n_splits}-fold family-grouped out-of-fold")
            for r, p in zip(train.itertuples(), oof)]
    for name in ["validation", "test"]:
        subset = joined[joined["split"] == name]
        pred = np.mean([m.predict(subset[features]) for m in fold_models], axis=0)
        rows += [dict(simulation_id=r.simulation_id, split=name, family_id=r.family_id,
                      actual_canopy_mean_m_s=float(getattr(r, TARGET)), predicted_canopy_mean_m_s=float(p),
                      prediction_method=f"mean of {n_splits} fold models (training families only)")
                 for r, p in zip(subset.itertuples(), pred)]
    table = pd.DataFrame(rows).sort_values("simulation_id").reset_index(drop=True)

    # Normalise with the TRUE training target statistics (same physical scale for every split).
    stats = {
        "mean": float(y_train.mean()),
        "std": float(y_train.std(ddof=0)) or 1.0,
        "fit_scope": "mean/std of the true canopy-mean target on training cases only",
    }
    table["channel_z"] = (table["predicted_canopy_mean_m_s"] - stats["mean"]) / stats["std"]

    diagnostics = []
    for name, part in table.groupby("split"):
        a, p = part["actual_canopy_mean_m_s"], part["predicted_canopy_mean_m_s"]
        diagnostics.append(dict(
            split=name, n_cases=len(part),
            corr_pred_vs_actual=float(np.corrcoef(a, p)[0, 1]),
            MAE_m_s=float(mean_absolute_error(a, p)),
            pred_std=float(p.std(ddof=0)),
            channel_z_min=float(part["channel_z"].min()), channel_z_max=float(part["channel_z"].max()),
            n_unique_predictions=int(p.round(5).nunique()),
        ))
    diagnostics = pd.DataFrame(diagnostics)

    test = table[table["split"] == "test"]
    scalar_metrics = {
        "test_cases": int(len(test)),
        "phase1_canopy_MAE_m_s": float(mean_absolute_error(test["actual_canopy_mean_m_s"], test["predicted_canopy_mean_m_s"])),
        "phase1_canopy_RMSE_m_s": float(np.sqrt(mean_squared_error(test["actual_canopy_mean_m_s"], test["predicted_canopy_mean_m_s"]))),
        "phase1_canopy_R2": float(r2_score(test["actual_canopy_mean_m_s"], test["predicted_canopy_mean_m_s"])),
    }

    table.to_csv(results_dir / "hybrid_phase1_canopy_predictions.csv", index=False)
    diagnostics.to_csv(results_dir / "hybrid_channel_diagnostics.csv", index=False)
    joblib.dump(fold_models, results_dir / "models" / "hybrid_phase1_xgboost_fold_models.joblib")
    (results_dir / "hybrid_canopy_channel_stats.json").write_text(json.dumps(stats, indent=2))
    (results_dir / "hybrid_phase1_scalar_test_metrics.json").write_text(json.dumps(scalar_metrics, indent=2))
    return table, stats, diagnostics, scalar_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Phase 1 XGBoost context channel.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root)
    _, _, diag, scalar = build_hybrid_channel(
        root / "data" / "processed" / "feature_engineered_dataset.csv",
        root / "dl_results" / "dl_split_manifest.csv",
        root / "dl_results",
    )
    print(diag.round(3).to_string(index=False))
    print(json.dumps(scalar, indent=2))
