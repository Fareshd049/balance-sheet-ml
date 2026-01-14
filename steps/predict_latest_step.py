from typing import Dict, Any, List
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from zenml import step

def _load_xy(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
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
) -> Dict[str, Any]:
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
        raise KeyError("No time column found (expected end/period_end_date/date).")

    df = df.dropna(subset=["cik", time_col]).sort_values(["cik", time_col])

    if mode == "latest_per_company":
        score_df = df.groupby("cik", as_index=False).tail(1).copy()
    else:
        score_df = df.copy()

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

    # also save a tiny run summary
    summary = {
        "timestamp": int(time.time()),
        "models_dir": models_dir,
        "dataset_scored": dataset_fe_path,
        "out_path": out_path,
        "rows_scored": int(len(preds)),
        "targets": used_targets,
        "mode": mode,
    }
    summary_path = str(Path(out_path).with_suffix(".json"))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[predict_latest_step] Saved predictions:", out_path)
    print("[predict_latest_step] Saved summary:", summary_path)

    return summary
