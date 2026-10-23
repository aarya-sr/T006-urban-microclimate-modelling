# UrbanTALES data and Phase 1 processing

## Phase 1 raw inputs

The first notebook, `Machine Learning Pipeline/ML_01_dataset_understanding_data_dictionary.ipynb`, is the canonical processing workflow. It reads:

- `data/raw/metadata_ideal.json` — 224 idealized cases and individual building-height arrays;
- `data/raw/metadata.csv` — the broader catalogue, used to cross-check simulation IDs;
- `data/raw/UrbanTALES-flowdata.csv` — raw flow statistics, including the `U` target for all 224 idealized cases.
- `data/raw/selected_files.zip` — sample vertical profile CSVs; ML_01 integrates each available sample profile up to mean building height and compares the result with the full-table `U` value as a diagnostic.
- `data/raw/selected_files (1).zip` — sample topology grids; ML_01 reads the grids and records dimensions/value summaries.
- `data/raw/selected_files (2).zip` — sample NetCDF files; ML_01 reads variable names and array shapes from both NetCDF4/HDF5 and classic NetCDF files.

The notebook builds geometry features from metadata, joins the target by simulation ID, checks the 224-case / 112-layout / 16-family hierarchy, validates missing values and duplicate IDs, and reads/audits all three supplied ZIP archives. It writes `data/processed/layout_level_dataset.csv`, `data_dictionary.csv`, `dataset_summary.json`, `raw_archive_manifest.csv`, `profile_target_crosscheck.csv`, `topology_archive_manifest.csv` and `netcdf_archive_manifest.csv`. The data dictionary is generated from the actual dataset columns and their documented meanings, and the same table is synchronized to `docs/data_dictionary.csv`. ML_03 separately writes `feature_engineered_dataset.csv` and `feature_engineered_data_dictionary.csv` after adding model features. It does not read the existing processed dataset or files in `results/`.

## Target definition

`target_U_canopy_mean_m_s` is taken from the raw `U` column in `UrbanTALES-flowdata.csv`. The supplied UrbanTALES datasheet describes the relevant flow-data values as canopy-averaged from ground level to mean building height and defines `U` as wind speed (m/s). This direct raw target is used so all 224 cases can be processed from the files included in the repository without relying on the partial sample of profile CSVs.

The profile ZIP contains only 32 sample profiles; the topology and NetCDF ZIPs also contain only 32 sample members each. ML_01 uses these to create diagnostics and file inventories, not to fill in unavailable targets or imply complete coverage. The profile-derived mean and flow-table `U` values are compared in `profile_target_crosscheck.csv`; the flow-table `U` remains the consistent scalar target for all 224 cases. Do not treat these sample files as a complete Phase 2 spatial-field dataset.

## Validation groups and labels

- `layout_id` is the folder stem before `_d00` or `_d45`.
- `family_id` is the plan family (arrangement letter plus plan-area-density code). All sibling layouts and both wind directions from a family must stay in the same fold for the primary validation protocol.
- `configuration` follows the folder-code convention (uniform / CL / HR / SM). The raw catalogue's `Config` label has CL and SM swapped relative to the folder code in the supplied metadata; the original value is preserved in `configuration_metadata_label` and excluded from model predictors.

## Run

Open and run the eight notebooks in `Machine Learning Pipeline/` in numerical order (ML_00 is the literature/problem-framing notebook). ML_01 builds the processed dataset; subsequent notebooks calculate EDA, fit models, compute evaluation metrics and create figures directly from code. The `results/` folder is an output destination for freshly computed tables, not an input. Plots are generated and displayed inline in the notebooks.

The supplied extraction notes state the UrbanTALES data licence as CC BY-NC 4.0. Retain dataset attribution and verify the original dataset terms before redistribution or commercial use. The dataset licence does not automatically apply to project code.
