

from typing import Dict, Any
from zenml import pipeline
from steps.add_ratio_features_step import add_ratio_features_step


@pipeline
def feature_ratio_pipeline(
    dataset_path: str,
    out_path: str,
) -> Dict[str, Any]:
    out_path2, stats = add_ratio_features_step(dataset_path=dataset_path, out_path=out_path)
    return {"out_path": out_path2, "stats": stats}
