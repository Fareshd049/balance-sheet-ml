from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd
from zenml import step

def _load_xy(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df.copy()
    for c in features:
        if c not in X.columns:
            X[c] = 0.0
    return X[features]

@step(enable_cache=False)
def predict_latest_step(
    dataset_fe_path: str,
    models_dir: str,
    out_path: str,
    mode: str = "latest_per_company",
    min_end: Optional[str] = None,
    year_filter: Optional[int] = None,
) -> Dict[str, Any]:
    df = pd.read_parquet(dataset_fe_path)

    time_col = "end" if "end" in df.columns else None
    if time_col is None:
        raise ValueError("No 'end' column found in dataset.")

    # robust datetime coercion BEFORE dropna
    if df[time_col].dtype == "object":
        df[time_col] = df[time_col].astype(str).str.strip()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    initial_rows = len(df)
    df = df.dropna(subset=["cik", time_col]).copy()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"[predict_latest_step] Dropped {dropped_rows} rows with missing cik or invalid dates")

    if len(df) == 0:
        raise ValueError(
            f"All {initial_rows} rows have missing or invalid dates in '{time_col}'. "
            "Check parse_company_facts_json_step output."
        )

    df = df.sort_values(["cik", time_col]).copy()
    print(f"[predict_latest_step] Date range: {df[time_col].min()} to {df[time_col].max()}")

    # Filtering (optional)
    if year_filter is not None:
        before = len(df)
        df = df[df[time_col].dt.year == int(year_filter)].copy()
        print(f"[predict_latest_step] year_filter={year_filter}: {before}->{len(df)}")
    elif min_end is not None:
        before = len(df)
        min_dt = pd.to_datetime(min_end)
        df = df[df[time_col] >= min_dt].copy()
        print(f"[predict_latest_step] min_end={min_end}: {before}->{len(df)}")

    if len(df) == 0:
        raise ValueError(
            "No rows remain after filtering. Lower min_end/year_filter."
        )

    # choose scoring rows
    if mode == "latest_per_company":
        score_df = df.groupby("cik", as_index=False).tail(1).copy()
        print(f"[predict_latest_step] Scoring latest per company: {len(score_df)}")
    else:
        score_df = df.copy()
        print(f"[predict_latest_step] Scoring all rows: {len(score_df)}")

    # load models
    model_files = sorted(Path(models_dir).glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No .joblib models in {models_dir}")

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
    with open(str(Path(out_path).with_suffix(".json")), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
