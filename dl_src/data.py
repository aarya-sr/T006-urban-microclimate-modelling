"""Dataset audit, raster preparation, grouped splitting and PyTorch data loading.

The UrbanTALES pedestrian NetCDF arrays are stored with dimensions (x, y).
For PyTorch and plotting, every spatial array is transposed to (y, x). The
original grid spacing is retained; cases are NOT resized to a common shape.
"""
from __future__ import annotations

import io
import json
import math
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PHASE1_NUMERIC_FEATURES = [
    "plan_area_density", "plan_area_density_sq", "height_std_m", "height_max_m",
    "height_min_m", "height_range_m", "height_skew", "frac_taller_than_mean",
    "wind_direction_sin", "wind_direction_cos", "height_coefficient_variation",
]
PHASE1_CATEGORICAL_FEATURES = [
    "configuration", "horizontal_configuration", "density_class",
]


def _family_id(layout_id):
    match = re.match(r"^[UV]([AS]\d+)", str(layout_id))
    if not match:
        raise ValueError(f"Cannot derive family_id from layout identifier: {layout_id}")
    return match.group(1)


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _decode_attr(value.reshape(-1)[0])
    return value


def build_case_manifest(metadata_json_path, metadata_csv_path, flowdata_csv_path):
    """Join the 224 idealized cases to catalogue conditions without using targets as inputs."""
    with open(metadata_json_path, "r", encoding="utf-8") as file:
        ideal_metadata = json.load(file)
    catalogue = pd.read_csv(metadata_csv_path)
    flowdata = pd.read_csv(flowdata_csv_path)

    tau_columns = [column for column in catalogue.columns if "tau" in column.lower()]
    if not tau_columns:
        raise KeyError("The catalogue does not contain the prescribed friction-velocity column.")
    tau_column = tau_columns[0]

    catalogue = catalogue.drop_duplicates(subset=["NameE"]).set_index("NameE")
    flow_lookup = flowdata.drop_duplicates(subset=["NameE"]).set_index("NameE")

    rows = []
    for item in ideal_metadata:
        case_id = str(item["Folder Name"])
        layout_id = case_id.rsplit("_d", 1)[0]
        heights = np.asarray(item.get("Heights", []), dtype=np.float32)
        if heights.size == 0:
            raise ValueError(f"No building-height list found for {case_id}.")
        if case_id not in catalogue.index:
            raise KeyError(f"{case_id} is missing from metadata.csv.")

        catalogue_row = catalogue.loc[case_id]
        flow_row = flow_lookup.loc[case_id] if case_id in flow_lookup.index else None
        wind_degrees = float(item["Wind Direction"])
        wind_radians = math.radians(wind_degrees)

        rows.append({
            "simulation_id": case_id,
            "layout_id": layout_id,
            "family_id": _family_id(layout_id),
            "wind_direction_deg": wind_degrees,
            "wind_direction_sin": math.sin(wind_radians),
            "wind_direction_cos": math.cos(wind_radians),
            "u_tau_m_s": float(catalogue_row[tau_column]),
            "dxdy_m": float(catalogue_row["dxdy"]),
            "plan_area_density": float(item["areaDensity"]),
            "configuration": str(item.get("Config", "unknown")),
            "horizontal_configuration": str(item.get("Horizontal Configuration", "unknown")),
            "height_mean_m": float(np.mean(heights)),
            "height_std_m": float(np.std(heights)),
            "height_min_m": float(np.min(heights)),
            "height_max_m": float(np.max(heights)),
            "building_count": int(heights.size),
            "domain_size_catalogue": str(catalogue_row.get("x-y", "")),
            "pedU_catalogue_m_s_diagnostic_only": (
                float(flow_row["pedU"]) if flow_row is not None and pd.notna(flow_row.get("pedU")) else np.nan
            ),
        })

    manifest = pd.DataFrame(rows).sort_values("simulation_id").reset_index(drop=True)
    if len(manifest) != 224:
        raise ValueError(f"Expected 224 idealized cases in metadata_ideal.json; found {len(manifest)}.")
    if manifest["simulation_id"].nunique() != len(manifest):
        raise ValueError("Duplicate simulation IDs found in the idealized metadata.")
    if manifest["layout_id"].nunique() != 112:
        raise ValueError(f"Expected 112 unique layouts; found {manifest['layout_id'].nunique()}.")
    if manifest["family_id"].nunique() != 16:
        raise ValueError(f"Expected 16 plan families; found {manifest['family_id'].nunique()}.")
    return manifest


def _find_member(names, expected_suffix):
    matches = [name for name in names if name.endswith(expected_suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ZIP member ending in {expected_suffix!r}; found {len(matches)}."
        )
    return matches[0]


def _read_pedestrian_netcdf(raw_bytes):
    """Read the NetCDF4/HDF5 datasets using h5py; netCDF4 is not required."""
    with h5py.File(io.BytesIO(raw_bytes), "r") as dataset:
        required = ["uped", "vped", "x", "y"]
        missing = [name for name in required if name not in dataset]
        if missing:
            raise KeyError(f"Pedestrian NetCDF is missing required datasets: {missing}")

        u = np.asarray(dataset["uped"][()], dtype=np.float32)
        v = np.asarray(dataset["vped"][()], dtype=np.float32)
        x = np.asarray(dataset["x"][()], dtype=np.float32)
        y = np.asarray(dataset["y"][()], dtype=np.float32)
        u_units = _decode_attr(dataset["uped"].attrs.get("units", None))
        v_units = _decode_attr(dataset["vped"].attrs.get("units", None))
        if u.ndim != 2 or v.ndim != 2:
            raise ValueError("uped and vped must both be two-dimensional spatial arrays.")
        if u.shape != v.shape:
            raise ValueError(f"Velocity component shapes differ: uped={u.shape}, vped={v.shape}.")
        if len(x) != u.shape[0] or len(y) != u.shape[1]:
            raise ValueError(
                f"Coordinate lengths do not match NetCDF dimensions: x={len(x)}, y={len(y)}, field={u.shape}."
            )
    return u, v, x, y, u_units, v_units


def make_grouped_split(manifest, test_size_families=3, validation_size_families=3, random_state=42):
    """Create deterministic train/validation/test splits by entire plan family.

    This is stricter than the minimum requirement of holding out complete layouts:
    every layout and both wind-direction cases belonging to a family remain together.
    """
    from sklearn.model_selection import GroupShuffleSplit

    family_table = manifest[["family_id"]].drop_duplicates().sort_values("family_id").reset_index(drop=True)
    families = family_table["family_id"].to_numpy()
    if len(families) != 16:
        raise ValueError("The grouped split expects 16 families from the idealized subset.")
    if test_size_families >= len(families) or validation_size_families >= len(families) - test_size_families:
        raise ValueError("Invalid requested number of held-out families.")

    first_split = GroupShuffleSplit(n_splits=1, test_size=test_size_families, random_state=random_state)
    train_val_family_idx, test_family_idx = next(first_split.split(family_table, groups=families))
    train_val_families = families[train_val_family_idx]
    test_families = set(families[test_family_idx])

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size_families,
        random_state=random_state + 1,
    )
    train_family_idx, val_family_idx = next(
        second_split.split(train_val_families, groups=train_val_families)
    )
    train_families = set(train_val_families[train_family_idx])
    val_families = set(train_val_families[val_family_idx])

    if train_families & val_families or train_families & test_families or val_families & test_families:
        raise AssertionError("Family overlap detected between splits.")

    rows = []
    for _, row in manifest.iterrows():
        family = row["family_id"]
        if family in train_families:
            split_name = "train"
        elif family in val_families:
            split_name = "validation"
        else:
            split_name = "test"
        rows.append({
            "simulation_id": row["simulation_id"],
            "layout_id": row["layout_id"],
            "family_id": family,
            "split": split_name,
        })

    split_manifest = pd.DataFrame(rows)
    layout_split_counts = split_manifest.groupby("layout_id")["split"].nunique()
    if not layout_split_counts.eq(1).all():
        raise AssertionError("At least one layout has been divided between splits.")
    family_split_counts = split_manifest.groupby("family_id")["split"].nunique()
    if not family_split_counts.eq(1).all():
        raise AssertionError("At least one family has been divided between splits.")
    return split_manifest


def prepare_phase2_dataset(archive_path, manifest, fields_dir, results_dir):
    """Audit all 224 cases and write compressed, model-ready field arrays.

    Each saved case contains height_map, u, v, valid_mask, x_m and y_m.
    The field arrays remain at their original 0.5 m grid resolution. They are
    transposed from (x, y) to (y, x), not resized or interpolated.
    """
    archive_path = Path(archive_path)
    fields_dir = Path(fields_dir)
    results_dir = Path(results_dir)
    fields_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    units_seen = {"uped": set(), "vped": set()}
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Corrupt ZIP member found: {bad_member}")
        names = archive.namelist()
        # The complete filtered archive is expected to contain all four requested file types.
        archive_coverage = {
            "pedestrian_netcdf": sum(name.endswith("_ped.nc") for name in names),
            "topography": sum(name.endswith("_topo") for name in names),
            "time_series_netcdf": sum(name.endswith("_ts.nc") for name in names),
            "profile_csv": sum(name.rsplit("/", 1)[-1].startswith("profile-") and name.endswith(".csv") for name in names),
        }
        if any(count != len(manifest) for count in archive_coverage.values()):
            raise ValueError(
                f"The full Phase 2 archive must contain {len(manifest)} members of each expected type; "
                f"observed coverage: {archive_coverage}"
            )

        for index, row in manifest.iterrows():
            case_id = row["simulation_id"]
            ped_member = _find_member(names, f"{case_id}_ped.nc")
            topo_member = _find_member(names, f"{case_id}_topo")
            ts_member = _find_member(names, f"{case_id}_ts.nc")
            profile_member = _find_member(names, f"profile-{case_id}.csv")

            u, v, x, y, u_units, v_units = _read_pedestrian_netcdf(archive.read(ped_member))
            topology = np.loadtxt(io.BytesIO(archive.read(topo_member)), dtype=np.float32)
            if topology.ndim != 2 or topology.shape != u.shape:
                raise ValueError(
                    f"Grid mismatch for {case_id}: topology={topology.shape}, velocity={u.shape}."
                )
            if not np.isfinite(topology).all():
                raise ValueError(f"Non-finite topography values found for {case_id}.")

            # Topography uses zero for open/air cells and positive height in metres for buildings.
            # A target cell is used only if it is open space and both velocity components are finite.
            valid_mask = (topology == 0) & np.isfinite(u) & np.isfinite(v)
            if int(valid_mask.sum()) == 0:
                raise ValueError(f"No valid pedestrian-level air cells found for {case_id}.")

            # NetCDF dimensions are (x, y); tensors and imshow arrays use (y, x).
            height_map = topology.T.astype(np.float32, copy=False)
            u_field = u.T.astype(np.float32, copy=False)
            v_field = v.T.astype(np.float32, copy=False)
            valid_mask_yx = valid_mask.T.astype(np.uint8, copy=False)
            x_m = (x * float(row["dxdy_m"])).astype(np.float32)
            y_m = (y * float(row["dxdy_m"])).astype(np.float32)

            output_path = fields_dir / f"{case_id}.npz"
            np.savez_compressed(
                output_path,
                height_map=height_map,
                u=u_field,
                v=v_field,
                valid_mask=valid_mask_yx,
                x_m=x_m,
                y_m=y_m,
            )

            units_seen["uped"].add(str(u_units))
            units_seen["vped"].add(str(v_units))
            audit_rows.append({
                "simulation_id": case_id,
                "pedestrian_netcdf_member": ped_member,
                "topography_member": topo_member,
                "time_series_member_verified_not_used": ts_member,
                "profile_member_verified_not_used": profile_member,
                "grid_x": int(u.shape[0]),
                "grid_y": int(u.shape[1]),
                "tensor_height_y": int(u.shape[1]),
                "tensor_width_x": int(u.shape[0]),
                "valid_air_cells": int(valid_mask.sum()),
                "building_cells": int((topology > 0).sum()),
                "nonfinite_velocity_cells": int((~(np.isfinite(u) & np.isfinite(v))).sum()),
                "max_building_height_m": float(np.max(topology)),
                "mean_stored_speed_on_valid_cells": float(np.mean(np.sqrt(u[valid_mask] ** 2 + v[valid_mask] ** 2))),
                "pedU_catalogue_m_s_diagnostic_only": float(row["pedU_catalogue_m_s_diagnostic_only"]),
                "dxdy_m": float(row["dxdy_m"]),
                "domain_x_length_m": float((x[-1] - x[0] + 1) * row["dxdy_m"]),
                "domain_y_length_m": float((y[-1] - y[0] + 1) * row["dxdy_m"]),
                "uped_units_attribute": str(u_units),
                "vped_units_attribute": str(v_units),
                "processed_file": str(output_path.relative_to(results_dir.parent)),
            })

    audit = pd.DataFrame(audit_rows)
    if len(audit) != len(manifest) or audit["simulation_id"].nunique() != len(manifest):
        raise AssertionError("Not every metadata case was successfully prepared exactly once.")

    audit_path = results_dir / "dl_data_manifest.csv"
    audit.to_csv(audit_path, index=False)
    audit[[
        "simulation_id", "mean_stored_speed_on_valid_cells", "pedU_catalogue_m_s_diagnostic_only",
        "uped_units_attribute", "vped_units_attribute"
    ]].to_csv(results_dir / "pedestrian_field_scalar_crosscheck.csv", index=False)
    summary = {
        "n_cases": int(len(audit)),
        "n_layouts": int(manifest["layout_id"].nunique()),
        "n_families": int(manifest["family_id"].nunique()),
        "archive_member_coverage": archive_coverage,
        "wind_directions_deg": sorted(manifest["wind_direction_deg"].unique().tolist()),
        "grid_shapes_xy": sorted({f"{r.grid_x}x{r.grid_y}" for r in audit.itertuples()}),
        "min_valid_air_cells": int(audit["valid_air_cells"].min()),
        "median_valid_air_cells": float(audit["valid_air_cells"].median()),
        "max_valid_air_cells": int(audit["valid_air_cells"].max()),
        "maximum_building_height_m": float(audit["max_building_height_m"].max()),
        "uped_units_attributes_observed": sorted(units_seen["uped"]),
        "vped_units_attributes_observed": sorted(units_seen["vped"]),
        "field_units_warning": (
            "The inspected NetCDF variables do not carry units attributes. Raw stored values are retained; "
            "do not label field errors as m/s unless the original data documentation confirms the units."
        ),
        "mask_definition": "topography == 0 AND finite(uped) AND finite(vped)",
        "orientation": "source arrays (x,y) transposed to model arrays (y,x)",
        "resampling": "none; original grid spacing retained",
    }
    with open(results_dir / "dl_dataset_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return audit, summary


def create_training_normalization(manifest, split_manifest, fields_dir, output_path):
    """Fit target and condition normalization using training families only."""
    train_ids = set(split_manifest.loc[split_manifest["split"] == "train", "simulation_id"])
    train_rows = manifest[manifest["simulation_id"].isin(train_ids)].copy()
    if len(train_rows) == 0:
        raise ValueError("The training split is empty; cannot fit normalization statistics.")

    count = 0
    sum_u = 0.0
    sum_v = 0.0
    sum_u2 = 0.0
    sum_v2 = 0.0
    max_height = 0.0

    for _, row in train_rows.iterrows():
        path = Path(fields_dir) / f"{row['simulation_id']}.npz"
        with np.load(path) as data:
            u = data["u"]
            v = data["v"]
            mask = data["valid_mask"].astype(bool)
            u_valid = u[mask].astype(np.float64)
            v_valid = v[mask].astype(np.float64)
            count += len(u_valid)
            sum_u += float(u_valid.sum())
            sum_v += float(v_valid.sum())
            sum_u2 += float(np.square(u_valid).sum())
            sum_v2 += float(np.square(v_valid).sum())
            max_height = max(max_height, float(data["height_map"].max()))

    mean_u = sum_u / count
    mean_v = sum_v / count
    std_u = math.sqrt(max(sum_u2 / count - mean_u ** 2, 1e-12))
    std_v = math.sqrt(max(sum_v2 / count - mean_v ** 2, 1e-12))
    tau_values = train_rows["u_tau_m_s"].astype(float).to_numpy()
    tau_mean = float(tau_values.mean())
    tau_std = float(tau_values.std(ddof=0))
    if tau_std < 1e-8:
        tau_std = 1.0

    stats = {
        "u_mean": mean_u,
        "u_std": std_u,
        "v_mean": mean_v,
        "v_std": std_v,
        "u_tau_mean": tau_mean,
        "u_tau_std": tau_std,
        "height_scale_m": max(max_height, 1.0),
        "training_valid_pixel_count": int(count),
        "normalization_fit_scope": "training split only",
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    return stats


class FlowFieldDataset(Dataset):
    """Lazy-loading PyTorch dataset for variable-size UrbanTALES spatial fields."""
    BASE_CHANNEL_NAMES = [
        "building_height_normalized",
        "wind_direction_sin",
        "wind_direction_cos",
        "friction_velocity_standardized",
        "domain_mask",
    ]

    def __init__(
        self,
        manifest,
        split_manifest,
        fields_dir,
        split,
        normalization_stats,
        channel_indices=None,
        canopy_predictions=None,
        canopy_prediction_stats=None,
    ):
        self.fields_dir = Path(fields_dir)
        self.stats = normalization_stats
        self.split_name = split
        self.canopy_predictions = canopy_predictions
        self.canopy_prediction_stats = canopy_prediction_stats
        self.channel_indices = channel_indices

        joined = manifest.merge(split_manifest[["simulation_id", "split"]], on="simulation_id", how="inner")
        self.rows = joined[joined["split"] == split].sort_values("simulation_id").reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No cases found for split {split!r}.")
        if channel_indices is not None:
            for index in channel_indices:
                if index < 0 or index >= 5:
                    raise ValueError("channel_indices must refer to the five base input channels.")

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _pad_spatial(array, pad_height, pad_width):
        if array.ndim == 2:
            return np.pad(array, ((0, pad_height), (0, pad_width)), mode="constant")
        return np.pad(array, ((0, 0), (0, pad_height), (0, pad_width)), mode="constant")

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        case_id = row["simulation_id"]
        field_path = self.fields_dir / f"{case_id}.npz"
        with np.load(field_path) as data:
            height_map = data["height_map"].astype(np.float32)
            u = data["u"].astype(np.float32)
            v = data["v"].astype(np.float32)
            valid_mask = data["valid_mask"].astype(np.float32)

        height, width = height_map.shape
        height_norm = height_map / float(self.stats["height_scale_m"])
        tau_standardized = (
            float(row["u_tau_m_s"]) - float(self.stats["u_tau_mean"])
        ) / float(self.stats["u_tau_std"])
        wind_sin = float(row["wind_direction_sin"])
        wind_cos = float(row["wind_direction_cos"])
        domain_mask = np.ones((height, width), dtype=np.float32)

        channels = [
            height_norm,
            np.full((height, width), wind_sin, dtype=np.float32),
            np.full((height, width), wind_cos, dtype=np.float32),
            np.full((height, width), tau_standardized, dtype=np.float32),
            domain_mask,
        ]
        if self.channel_indices is not None:
            channels = [channels[channel_index] for channel_index in self.channel_indices]

        if self.canopy_predictions is not None:
            if self.canopy_prediction_stats is None:
                raise ValueError("canopy_prediction_stats are required when canopy predictions are used.")
            prediction = float(self.canopy_predictions[case_id])
            prediction_z = (
                prediction - float(self.canopy_prediction_stats["mean"])
            ) / float(self.canopy_prediction_stats["std"])
            channels.append(np.full((height, width), prediction_z, dtype=np.float32))

        input_array = np.stack(channels, axis=0).astype(np.float32)
        target_array = np.stack([
            (u - float(self.stats["u_mean"])) / float(self.stats["u_std"]),
            (v - float(self.stats["v_mean"])) / float(self.stats["v_std"]),
        ], axis=0).astype(np.float32)
        # NaN multiplied by a zero mask is still NaN. Replace invalid target cells
        # with zero before the masked loss; their mask remains zero, so they add no loss.
        target_array[:, valid_mask == 0] = 0.0

        # The U-Net has four downsampling stages. Pad only at the bottom/right to a multiple of 16.
        pad_height = (16 - height % 16) % 16
        pad_width = (16 - width % 16) % 16
        input_array = self._pad_spatial(input_array, pad_height, pad_width)
        target_array = self._pad_spatial(target_array, pad_height, pad_width)
        valid_mask = self._pad_spatial(valid_mask, pad_height, pad_width)

        return (
            torch.from_numpy(input_array),
            torch.from_numpy(target_array),
            torch.from_numpy(valid_mask[None, :, :]),
            case_id,
            torch.tensor([height, width], dtype=torch.int64),
        )
