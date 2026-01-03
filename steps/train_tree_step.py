from typing import Tuple, List
from pathlib import Path
import joblib
import pandas as pd
from zenml import step

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

# keep the same RATIO_COLS list as in train_baseline_step.py
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
    lag_feats = sorted(c for c in df.columns if c.endswith("_lag1"))
    ratio_base = [c for c in RATIO_COLS if c in df.columns]

    derived_ratio_feats: List[str] = []
    if ratio_base:
        prefixes = tuple(r + "_" for r in ratio_base)
        derived_ratio_feats = sorted(
            c for c in df.columns if c.startswith(prefixes) and c not in ratio_base
        )

    extra_ratio_feats = [
        c for c in ["cash_to_current_liab", "current_ratio", "net_debt_to_assets"]
        if c in df.columns
    ]

    flag = f"{target_col}_is_log"
    islog = [flag] if flag in df.columns else []

    feats = lag_feats + ratio_base + derived_ratio_feats + extra_ratio_feats + islog
    drop = {"cik", "end", target_col}
    feats = [c for c in feats if c not in drop]

    out = []
    for c in feats:
        if c not in out:
            out.append(c)
    return out


@step(enable_cache=False)
def train_tree_step(
    train_path: str,
    target_col: str,
    max_iter: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    early_stopping: bool = True,
    n_iter_no_change: int = 20,
    validation_fraction: float = 0.1,
) -> Tuple[str, str]:
    df = pd.read_parquet(train_path).dropna(subset=[target_col]).copy()

    feature_cols = _pick_features(df, target_col)
    X = df[feature_cols]
    y = df[target_col].astype("float64").values

    hgb = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
        validation_fraction=validation_fraction,
        random_state=42,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("hgb", hgb),
    ])

    model.fit(X, y)

    bundle = {"model": model, "features": feature_cols, "target": target_col}

    out_dir = Path("data/interim/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    # include params in filename so runs don't overwrite each other
    model_path = str(
        out_dir / f"hgb_{target_col}_it{max_iter}_lr{learning_rate}_d{max_depth}.joblib"
    )
    joblib.dump(bundle, model_path)

    train_stats = {
        "rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "model": "HistGradientBoostingRegressor",
        "max_iter": int(max_iter),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "early_stopping": bool(early_stopping),
        "n_iter_no_change": int(n_iter_no_change),
        "validation_fraction": float(validation_fraction),
        "model_path": model_path,
    }
    train_stats_json = pd.Series([train_stats]).to_json(orient="records")

    # helpful debug
    print("[train_tree_step] N FEATURES:", len(feature_cols))
    print("[train_tree_step] params:", train_stats)

    return model_path, train_stats_json

