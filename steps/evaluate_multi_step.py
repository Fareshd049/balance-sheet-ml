from typing import Dict, Any
import time
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from zenml import step
from sklearn.inspection import permutation_importance


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    dir_acc = float(np.mean((y_true >= 0) == (y_pred >= 0)))
    return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc}


def _load_xy(df: pd.DataFrame, features: list[str], target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [c for c in features if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = np.nan
        print(f"[evaluate_multi_step] Added missing cols: {len(missing)}")

    X = df.loc[:, features].astype("float32")
    y = df[target_col].astype("float64").values
    return X, y


@step(enable_cache=False)
def evaluate_multi_step(
    model_paths: Dict[str, str],   # target_col -> model_path
    val_path: str,
    test_path: str,
    compute_importance: bool = False,   # default False to keep runtime reasonable
    importance_repeats: int = 5,
    importance_topk: int = 30,
) -> Dict[str, Any]:

    results: Dict[str, Any] = {}

    for target_col, model_path in model_paths.items():
        bundle = joblib.load(model_path)
        model = bundle["model"]
        features = bundle["features"]

        def eval_split(path: str) -> Dict[str, Any]:
            df = pd.read_parquet(path).dropna(subset=[target_col]).copy()
            if len(df) == 0:
                return {"rows": 0, "mae": None, "rmse": None, "dir_acc": None}
            X, y = _load_xy(df, features, target_col)
            pred = model.predict(X)
            m = _metrics(y, pred)
            m["rows"] = int(len(df))
            return m

        out = {"val": eval_split(val_path), "test": eval_split(test_path)}

        # Optional: permutation importance per target (can be slow)
        if compute_importance:
            dfv = pd.read_parquet(val_path).dropna(subset=[target_col]).copy()
            if len(dfv) > 0:
                Xv, yv = _load_xy(dfv, features, target_col)
                r = permutation_importance(
                    model,
                    Xv,
                    yv,
                    n_repeats=int(importance_repeats),
                    random_state=42,
                    scoring="neg_root_mean_squared_error",
                )
                imp = pd.DataFrame({"feature": features, "importance": r.importances_mean}) \
                        .sort_values("importance", ascending=False)
                out["val_importance_top"] = imp.head(int(importance_topk)).to_dict(orient="records")

        results[target_col] = out

    # Save one combined summary
    Path("data/interim/metrics").mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": int(time.time()), "metrics_by_target": results}
    fname = f"data/interim/metrics/metrics_multi_{payload['timestamp']}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[evaluate_multi_step] Saved metrics to: {fname}")
    return results
