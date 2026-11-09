"""Path helpers for the T006 Phase 2 pipeline."""
from pathlib import Path


def find_project_root(start=None):
    """Find the existing T006 repository root without assuming a fixed cwd."""
    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "data" / "raw").is_dir() and (candidate / "Machine Learning Pipeline").is_dir():
            return candidate
        # In Colab, the repository is commonly cloned as /content/<repo-name>.
        if candidate.is_dir():
            for child in candidate.iterdir():
                if child.is_dir() and (child / "data" / "raw").is_dir() and (child / "Machine Learning Pipeline").is_dir():
                    return child
    raise FileNotFoundError(
        "Could not locate the T006 repository root. Run this notebook from inside "
        "the existing T006-urban-microclimate-modelling repository."
    )


def get_project_paths(start=None):
    """Return the standard project paths and create only the runtime output folders."""
    root = find_project_root(start)
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    dl_results = root / "dl_results"
    fields = dl_results / "processed_fields"
    figures = dl_results / "figures"
    models = dl_results / "models"
    predictions = dl_results / "predictions"

    for folder in [dl_results, fields, figures, models, predictions]:
        folder.mkdir(parents=True, exist_ok=True)

    archive_candidates = [
        raw / "selected_files (4).zip",
        raw / "selected_files_phase2.zip",
        raw / "selected_files_phase2_full.zip",
        raw / "selected_files_rejoined.zip",
    ]
    archive = next((path for path in archive_candidates if path.exists()), None)

    return {
        "root": root,
        "raw": raw,
        "processed": processed,
        "dl_results": dl_results,
        "fields": fields,
        "figures": figures,
        "models": models,
        "predictions": predictions,
        "archive": archive,
    }
