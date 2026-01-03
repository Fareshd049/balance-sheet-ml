from typing import Dict, Any
from pathlib import Path
import json
import time

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
        print(f"[evaluate_step] Added missing cols: {len(missing)}")

    X = df.loc[:, features].astype("float32")
    y = df[target_col].astype("float64").values
    return X, y


@step(enable_cache=False)
def evaluate_step(
    model_path: str,
    val_path: str,
    test_path: str,
    compute_importance: bool = True,
    importance_repeats: int = 5,
    importance_topk: int = 30,
) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    target_col = bundle["target"]

    print(f"[evaluate_step] Model expects {len(features)} features")

    def eval_split(path: str) -> Dict[str, Any]:
        df = pd.read_parquet(path).dropna(subset=[target_col]).copy()
        X, y = _load_xy(df, features, target_col)

        pred = model.predict(X)
        m = _metrics(y, pred)
        m["rows"] = int(len(df))
        return m

    out: Dict[str, Any] = {
        "val": eval_split(val_path),
        "test": eval_split(test_path),
    }

    # --- Permutation importance on VAL (optional) ---
    if compute_importance:
        dfv = pd.read_parquet(val_path).dropna(subset=[target_col]).copy()
        Xv, yv = _load_xy(dfv, features, target_col)

        print("\n[evaluate_step] Computing permutation importance on VAL...")
        r = permutation_importance(
            model,
            Xv,
            yv,
            n_repeats=int(importance_repeats),
            random_state=42,
            scoring="neg_root_mean_squared_error",
        )
        imp = pd.DataFrame(
            {"feature": features, "importance": r.importances_mean}
        ).sort_values("importance", ascending=False)

        top = imp.head(int(importance_topk)).copy()
        print("\nTop permutation importances (VAL):")
        print(top.to_string(index=False))

        # store as list of dicts (JSON-friendly)
        out["val_importance_top"] = top.to_dict(orient="records")

    print("\n📊 METRICS:", out)

    # --- Save run summary to disk (easy run-to-run comparison) ---
    Path("data/interim/metrics").mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": int(time.time()),
        "model_path": model_path,
        "target": target_col,
        "n_features": len(features),
        "metrics": out,
    }
    fname = f"data/interim/metrics/metrics_{payload['timestamp']}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[evaluate_step] Saved metrics to: {fname}")
    return out
