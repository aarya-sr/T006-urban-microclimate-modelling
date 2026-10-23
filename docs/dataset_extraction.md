# UrbanTALES profile extraction

1. `src/download_dataset.py` retrieves the `metadata_ideal.json` catalogue from the UrbanTALES API if it is not present.
2. For each of the 224 idealized cases, it selects the direct link for `profile-<case>.csv` and downloads the small vertical profile file into `data/profiles/`.
3. `src/build_layout_dataset.py` joins the supplied profile column `U` with catalogue geometry metadata and creates `data/processed/layout_level_dataset.csv`.
4. The target is defined explicitly as the trapezoidal height-weighted mean of `U(z)` over `0 <= z <= mean building height`, with linear interpolation at the endpoints.
5. The layout key is the folder stem before `_d00` or `_d45`. The **validation group is the plan family** (`family_id`): the arrangement letter plus density code of the layout (for example `A35` from `VA35-HR-28` and `UA35`). All 7 sibling layouts and both wind directions of a family stay together in every fold.
6. `configuration` follows the folder code (uniform / CL / HR / SM). The catalogue `Config` field has CL and SM swapped relative to the folder code; the raw value is stored as `configuration_metadata_label` and never used as a predictor.
7. Without the raw profiles, `src/refresh_features.py` rebuilds every predictor and the labels from `metadata_ideal.json` and keeps the stored target.

Run from the repository root:

```bash
cd src
python download_dataset.py
python build_layout_dataset.py
```

`profile-*.csv` files are source data and are excluded from the repository archive; the 224-row final merged dataset is included. Retain the dataset attribution and CC BY-NC 4.0 license notice from UrbanTALES. The metadata has wind direction but no separate weather time-series variables or documented per-case inlet-speed field used in this project. Do not infer or fabricate those inputs.

## Target interpretation

The profile CSV provides `z` and `U`. We use the file's `U` values as provided and do not relabel the target as pedestrian-level wind, maximum speed, or a full 3-D domain mean. The operationally defined canopy-layer profile average supports a scalar regression task consistent with Phase 1; spatial field prediction is outside this package.
