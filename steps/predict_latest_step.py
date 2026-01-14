from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from zenml import step


def _load_xy(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Ensure all expected feature columns exist and return float32 matrix."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = np.nan
        print(f"[predict_latest_step] Added missing cols: {len(missing)}")
    return df.loc[:, features].astype("float32")


@step(enable_cache=False)
def predict_latest_step(
    dataset_fe_path: str,
    models_dir: str,
    out_path: str,
    mode: str = "latest_per_company",
    # NEW: keep only rows at/after this date (so outputs are 2025/2026 only)
    min_end: Optional[str] = "2025-01-01",
    # Optional: hard filter to a single year (e.g., 2025). If set, overrides min_end.
    year_filter: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Score the dataset using all saved target models.

    - Reads engineered dataset (with ratio deltas/rolling stats).
    - Optionally filters to recent dates (min_end) or a single year (year_filter).
    - If mode='latest_per_company': scores only the latest row per cik (after filtering).
    - Saves predictions parquet + summary JSON.
    """
    df = pd.read_parquet(dataset_fe_path)

    if "cik" not in df.columns:
        raise KeyError("cik not found in dataset")

    # time column
    if "end" in df.columns:
        time_col = "end"
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
    elif "period_end_date" in df.columns:
        time_col = "period_end_date"
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        raise KeyError("No time column found (expected end/period_end_date).")

    df = df.dropna(subset=["cik", time_col]).sort_values(["cik", time_col]).copy()

    # -----------------------
    # NEW: date-based filtering
    # -----------------------
    if year_filter is not None:
        df = df[df[time_col].dt.year == int(year_filter)].copy()
        print(f"[predict_latest_step] Applied year_filter={year_filter}, rows={len(df)}")
    elif min_end is not None:
        min_dt = pd.to_datetime(min_end)
        df = df[df[time_col] >= min_dt].copy()
        print(f"[predict_latest_step] Applied min_end={min_end}, rows={len(df)}")

    if len(df) == 0:
        raise ValueError(
            f"No rows remain after filtering (min_end={min_end}, year_filter={year_filter}). "
            "Lower the cutoff or ensure the dataset contains recent filings."
        )

    # Choose what to score
    if mode == "latest_per_company":
        score_df = df.groupby("cik", as_index=False).tail(1).copy()
    else:
        score_df = df.copy()

    # Load models
    model_files = sorted(Path(models_dir).glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No .joblib models found in {models_dir}")

    preds = score_df[["cik", time_col]].copy()
    used_targets = []

    for mf in model_files:
        bundle = joblib.load(mf)
        model = bundle["model"]
        features = bundle["features"]
        target = bundle.get("target", mf.stem)

        X = _load_xy(score_df, features)
        yhat = model.predict(X)

        preds[f"pred_{target}"] = yhat
        used_targets.append(str(target))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_path, index=False)

    # Save run summary
    summary = {
        "timestamp": int(time.time()),
        "models_dir": models_dir,
        "dataset_scored": dataset_fe_path,
        "out_path": out_path,
        "rows_scored": int(len(preds)),
        "targets": used_targets,
        "mode": mode,
        "min_end": min_end,
        "year_filter": year_filter,
    }
    summary_path = str(Path(out_path).with_suffix(".json"))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[predict_latest_step] Saved predictions:", out_path)
    print("[predict_latest_step] Saved summary:", summary_path)

    return summary
