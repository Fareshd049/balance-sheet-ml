

from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from zenml import step

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def _pick_features(df: pd.DataFrame, target_col: str) -> List[str]:
    # baseline: use all lag1 columns + is_log flags (optional)
    lag_feats = [c for c in df.columns if c.endswith("_lag1")]
    islog_feats = [c for c in df.columns if c.startswith("y_") and c.endswith("_is_log")]
    # exclude target itself if it accidentally matches
    feats = [c for c in (lag_feats + islog_feats) if c != target_col]
    return feats


@step
def train_baseline_step(
    train_path: str,
    target_col: str = "y_Assets",
    model_out: str = "data/processed/models/ridge_y_assets.joblib",
    alpha: float = 1.0,
) -> Tuple[str, dict]:
    df = pd.read_parquet(train_path)

    features = _pick_features(df, target_col)
    X = df[features]
    y = df[target_col].astype(float)

    # Simple, robust baseline
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=alpha, random_state=42)),
        ]
    )

    model.fit(X, y)

    import os
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({"model": model, "features": features, "target": target_col}, model_out)

    return model_out, {
        "model_path": model_out,
        "n_train": int(len(df)),
        "n_features": int(len(features)),
        "alpha": float(alpha),
    }
