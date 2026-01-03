from typing import Dict, Any, List, Tuple
from pathlib import Path
import joblib
import pandas as pd
from zenml import step

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


RATIO_COLS = [
    "cash_to_assets",
    "current_assets_to_assets",
    "liab_to_assets",
    "equity_to_assets",
    "current_liab_to_liab",
    "ppe_to_assets",
    "goodwill_to_assets",
    "intangibles_to_assets",
    "wc_to_assets",
    "debt_to_assets",
    "debt_to_equity",
]


def _pick_features(df: pd.DataFrame, target_col: str) -> List[str]:
    # existing lag logic
    lag_feats = sorted(c for c in df.columns if c.endswith("_lag1"))

    # base ratios that exist
    ratio_base = [c for c in RATIO_COLS if c in df.columns]

    # derived ratio features from base ratios:
    # cash_to_assets_delta1, cash_to_assets_pct1, cash_to_assets_rollmean4, ...
    derived_ratio_feats: List[str] = []
    if ratio_base:
        prefixes = tuple(r + "_" for r in ratio_base)
        derived_ratio_feats = sorted(
            c for c in df.columns
            if c.startswith(prefixes) and c not in ratio_base
        )

    # optional extra ratios if present
    extra_ratio_feats = [
        c for c in ["cash_to_current_liab", "current_ratio", "net_debt_to_assets"]
        if c in df.columns
    ]

    # optional is_log flag
    flag = f"{target_col}_is_log"
    islog = [flag] if flag in df.columns else []

    feats = lag_feats + ratio_base + derived_ratio_feats + extra_ratio_feats + islog

    # keep your original drops
    drop = {"cik", "end", target_col}
    feats = [c for c in feats if c not in drop]

    # de-dup preserve order
    out: List[str] = []
    for c in feats:
        if c not in out:
            out.append(c)
    return out


@step(enable_cache=False)
def train_baseline_step(
    train_path: str,
    target_col: str,
    alpha: float = 1.0,
) -> Tuple[str, str]:
    df = pd.read_parquet(train_path).dropna(subset=[target_col]).copy()

    feature_cols = _pick_features(df, target_col)
    X = df[feature_cols]
    y = df[target_col].astype("float64").values

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(X, y)

    bundle = {"model": model, "features": feature_cols, "target": target_col}

    out_dir = Path("data/interim/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(out_dir / f"baseline_ridge_{target_col}.joblib")
    joblib.dump(bundle, model_path)

    train_stats = {
        "rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "alpha": float(alpha),
        "model_path": model_path,
    }

    train_stats_json = pd.Series([train_stats]).to_json(orient="records")

    print("[train_baseline_step] N FEATURES:", len(feature_cols))
    print("[train_baseline_step] N LAG:", sum(c.endswith("_lag1") for c in feature_cols))
    print("[train_baseline_step] N RATIOS:", sum(("_to_" in c) or c.endswith("_ratio") for c in feature_cols))
    print("[train_baseline_step] RATIOS:", [c for c in feature_cols if ("_to_" in c) or c.endswith("_ratio")])
    print("[train_baseline_step] IS_LOG:", [c for c in feature_cols if c.endswith("_is_log")])

    return model_path, train_stats_json
