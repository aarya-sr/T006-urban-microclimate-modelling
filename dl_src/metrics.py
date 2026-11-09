"""Field metrics, test-set inference, spatial visualizations and result tables."""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def calculate_field_metrics(u_true, v_true, u_pred, v_pred, valid_mask):
    """Calculate component and speed errors over valid pedestrian-level air cells."""
    mask = valid_mask.astype(bool)
    if int(mask.sum()) == 0:
        raise ValueError("Cannot calculate field metrics: valid mask contains no cells.")

    true_components = np.concatenate([u_true[mask], v_true[mask]])
    predicted_components = np.concatenate([u_pred[mask], v_pred[mask]])
    component_error = predicted_components - true_components
    component_mae = float(np.mean(np.abs(component_error)))
    component_rmse = float(np.sqrt(np.mean(component_error ** 2)))
    component_range = float(np.max(true_components) - np.min(true_components))
    component_nrmse = component_rmse / component_range if component_range > 0 else np.nan

    true_speed = np.sqrt(u_true[mask] ** 2 + v_true[mask] ** 2)
    predicted_speed = np.sqrt(u_pred[mask] ** 2 + v_pred[mask] ** 2)
    speed_error = predicted_speed - true_speed
    speed_mae = float(np.mean(np.abs(speed_error)))
    speed_rmse = float(np.sqrt(np.mean(speed_error ** 2)))
    speed_range = float(np.max(true_speed) - np.min(true_speed))
    speed_nrmse = speed_rmse / speed_range if speed_range > 0 else np.nan

    return {
        "field_component_MAE": component_mae,
        "field_component_RMSE": component_rmse,
        "field_component_NRMSE": component_nrmse,
        "pedestrian_speed_MAE_native_units": speed_mae,
        "pedestrian_speed_RMSE_native_units": speed_rmse,
        "pedestrian_speed_NRMSE": speed_nrmse,
        "valid_air_cells": int(mask.sum()),
    }


def evaluate_model(model, data_loader, normalization_stats, device, model_name, results_dir, fields_dir):
    """Run inference on every case in a loader and save per-case predictions and metrics."""
    results_dir = Path(results_dir)
    prediction_dir = results_dir / "predictions" / model_name
    figure_dir = results_dir / "figures" / model_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()
    case_rows = []
    prediction_cache = {}

    with torch.no_grad():
        for inputs, normalized_targets, masks, case_ids, original_shapes in data_loader:
            inputs = inputs.to(device)
            normalized_targets = normalized_targets.to(device)
            prediction_normalized = model(inputs).cpu().numpy()[0]
            target_normalized = normalized_targets.cpu().numpy()[0]
            mask_padded = masks.cpu().numpy()[0, 0]

            case_id = case_ids[0]
            original_height = int(original_shapes[0][0])
            original_width = int(original_shapes[0][1])

            prediction_normalized = prediction_normalized[:, :original_height, :original_width]
            target_normalized = target_normalized[:, :original_height, :original_width]
            mask = mask_padded[:original_height, :original_width].astype(bool)

            u_pred = prediction_normalized[0] * float(normalization_stats["u_std"]) + float(normalization_stats["u_mean"])
            v_pred = prediction_normalized[1] * float(normalization_stats["v_std"]) + float(normalization_stats["v_mean"])
            u_true = target_normalized[0] * float(normalization_stats["u_std"]) + float(normalization_stats["u_mean"])
            v_true = target_normalized[1] * float(normalization_stats["v_std"]) + float(normalization_stats["v_mean"])

            metrics = calculate_field_metrics(u_true, v_true, u_pred, v_pred, mask)
            metrics["simulation_id"] = case_id
            case_rows.append(metrics)

            with np.load(Path(fields_dir) / f"{case_id}.npz") as field_data:
                x_m = field_data["x_m"].copy()
                y_m = field_data["y_m"].copy()
                height_map = field_data["height_map"].copy()

            np.savez_compressed(
                prediction_dir / f"{case_id}_prediction.npz",
                u_true=u_true.astype(np.float32),
                v_true=v_true.astype(np.float32),
                u_pred=u_pred.astype(np.float32),
                v_pred=v_pred.astype(np.float32),
                valid_mask=mask.astype(np.uint8),
                height_map=height_map.astype(np.float32),
                x_m=x_m.astype(np.float32),
                y_m=y_m.astype(np.float32),
            )
            prediction_cache[case_id] = {
                "u_true": u_true,
                "v_true": v_true,
                "u_pred": u_pred,
                "v_pred": v_pred,
                "mask": mask,
                "height_map": height_map,
                "x_m": x_m,
                "y_m": y_m,
            }

    case_metrics = pd.DataFrame(case_rows).sort_values("simulation_id").reset_index(drop=True)
    # Macro average: every held-out case receives equal weight, regardless of grid size.
    numeric_columns = [column for column in case_metrics.columns if column != "simulation_id"]
    mean_case_rows = []
    for column in numeric_columns:
        value = int(case_metrics[column].sum()) if column == "valid_air_cells" else float(case_metrics[column].mean())
        mean_case_rows.append({"metric": column, "mean_across_cases": value})
    mean_case_metrics = pd.DataFrame(mean_case_rows)

    # Pooled field metrics: every valid air cell across all held-out cases is pooled.
    all_u_true = []
    all_v_true = []
    all_u_pred = []
    all_v_pred = []
    for prediction in prediction_cache.values():
        mask = prediction["mask"].astype(bool)
        all_u_true.append(prediction["u_true"][mask])
        all_v_true.append(prediction["v_true"][mask])
        all_u_pred.append(prediction["u_pred"][mask])
        all_v_pred.append(prediction["v_pred"][mask])
    pooled_u_true = np.concatenate(all_u_true)
    pooled_v_true = np.concatenate(all_v_true)
    pooled_u_pred = np.concatenate(all_u_pred)
    pooled_v_pred = np.concatenate(all_v_pred)
    pooled_mask = np.ones(len(pooled_u_true), dtype=bool)
    pooled_metrics_dict = calculate_field_metrics(
        pooled_u_true, pooled_v_true, pooled_u_pred, pooled_v_pred, pooled_mask
    )
    pooled_metrics = pd.DataFrame([pooled_metrics_dict])
    pooled_metrics.insert(0, "aggregation", "all valid cells pooled")

    case_metrics.to_csv(results_dir / f"{model_name}_per_case_metrics.csv", index=False)
    mean_case_metrics.to_csv(results_dir / f"{model_name}_mean_case_metrics.csv", index=False)
    pooled_metrics.to_csv(results_dir / f"{model_name}_pooled_field_metrics.csv", index=False)
    return case_metrics, pooled_metrics, prediction_cache


def save_field_visualization(case_id, prediction, output_path):
    """Save ground-truth, prediction and error maps for one held-out layout case."""
    mask = prediction["mask"].astype(bool)
    true_speed = np.sqrt(prediction["u_true"] ** 2 + prediction["v_true"] ** 2)
    predicted_speed = np.sqrt(prediction["u_pred"] ** 2 + prediction["v_pred"] ** 2)
    vector_error = np.sqrt(
        (prediction["u_pred"] - prediction["u_true"]) ** 2
        + (prediction["v_pred"] - prediction["v_true"]) ** 2
    )

    true_speed_plot = np.ma.masked_where(~mask, true_speed)
    predicted_speed_plot = np.ma.masked_where(~mask, predicted_speed)
    error_plot = np.ma.masked_where(~mask, vector_error)
    height_plot = np.ma.masked_where(prediction["height_map"] <= 0, prediction["height_map"])

    x = prediction["x_m"]
    y = prediction["y_m"]
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    image0 = axes[0].imshow(height_plot, origin="lower", extent=extent, aspect="auto")
    axes[0].set_title("Building height (m)")
    fig.colorbar(image0, ax=axes[0], fraction=0.046)

    image1 = axes[1].imshow(true_speed_plot, origin="lower", extent=extent, aspect="auto")
    axes[1].set_title("True pedestrian speed")
    fig.colorbar(image1, ax=axes[1], fraction=0.046)

    image2 = axes[2].imshow(predicted_speed_plot, origin="lower", extent=extent, aspect="auto")
    axes[2].set_title("Predicted pedestrian speed")
    fig.colorbar(image2, ax=axes[2], fraction=0.046)

    image3 = axes[3].imshow(error_plot, origin="lower", extent=extent, aspect="auto")
    axes[3].set_title("Vector error magnitude")
    fig.colorbar(image3, ax=axes[3], fraction=0.046)

    for axis in axes:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    fig.suptitle(f"Held-out case: {case_id}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.show()
    plt.close(fig)


def save_selected_error_maps(prediction_cache, output_dir, number_of_cases=3):
    """Choose representative cases by sorted ID and save their spatial diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = []
    selected_families = set()
    for case_id in sorted(prediction_cache.keys()):
        layout_id = case_id.rsplit("_d", 1)[0]
        match = re.match(r"^[UV]([AS]\d+)", layout_id)
        family_id = match.group(1) if match else layout_id
        if family_id not in selected_families:
            selected_ids.append(case_id)
            selected_families.add(family_id)
        if len(selected_ids) >= number_of_cases:
            break
    for case_id in selected_ids:
        save_field_visualization(
            case_id,
            prediction_cache[case_id],
            output_dir / f"{case_id}_field_comparison.png",
        )
    return selected_ids
