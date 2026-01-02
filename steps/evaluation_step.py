

from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from zenml import step

from sklearn.metrics import mean_absolute_error, mean_squared_error


@step
def evaluate_step(
    model_path: str,
    val_path: str,
    test_path: str,
) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    target = bundle["target"]

    def eval_split(path: str) -> Dict[str, float]:
        df = pd.read_parquet(path)
        df = df.dropna(subset=[target]).copy()
        X = df[features]
        y = df[target].astype(float).to_numpy()
        pred = model.predict(X)

        mae = float(mean_absolute_error(y, pred))
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        # Directional accuracy: sign(pred) == sign(y)
        dir_acc = float((np.sign(pred) == np.sign(y)).mean())

        return {"rows": int(len(df)), "mae": mae, "rmse": rmse, "dir_acc": dir_acc}

    val_metrics = eval_split(val_path)
    test_metrics = eval_split(test_path)

    return {
        "val": val_metrics,
        "test": test_metrics,
    }
