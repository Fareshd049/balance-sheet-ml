from typing import Dict, List, Tuple
from pathlib import Path
import joblib
import pandas as pd
from zenml import step

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

from steps.train_tree_step import _pick_features  # reuse your feature logic


@step(enable_cache=False)
def train_tree_multi_step(
    train_path: str,
    target_cols: List[str],
    max_iter: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    early_stopping: bool = True,
    n_iter_no_change: int = 20,
    validation_fraction: float = 0.1,
) -> Tuple[Dict[str, str], str]:
    df = pd.read_parquet(train_path)

    out_dir = Path("data/interim/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_paths: Dict[str, str] = {}
    stats_rows = []

    for target_col in target_cols:
        dft = df.dropna(subset=[target_col]).copy()

        # guardrail: skip targets with too little data
        if len(dft) < 200:
            print(f"[train_tree_multi_step] Skip {target_col}: too few rows ({len(dft)})")
            continue

        feature_cols = _pick_features(dft, target_col)
        X = dft[feature_cols]
        y = dft[target_col].astype("float64").values

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

        model_path = str(out_dir / f"hgb_{target_col}_it{max_iter}_lr{learning_rate}_d{max_depth}.joblib")
        joblib.dump(bundle, model_path)

        model_paths[target_col] = model_path
        stats_rows.append({
            "target": target_col,
            "rows": int(len(dft)),
            "n_features": int(len(feature_cols)),
            "model_path": model_path,
        })

        print(f"[train_tree_multi_step] Trained {target_col}: rows={len(dft)} feats={len(feature_cols)}")

    stats_json = pd.DataFrame(stats_rows).to_json(orient="records")
    return model_paths, stats_json
