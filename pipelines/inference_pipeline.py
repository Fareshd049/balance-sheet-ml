from typing import Dict, Any
from zenml import pipeline

from steps.score_feature_engineering_step import score_feature_engineering_step
from steps.predict_latest_step import predict_latest_step

@pipeline(enable_cache=False)
def inference_pipeline(
    dataset_path: str,
    models_dir: str,
    out_path: str,
    mode: str = "latest_per_company",
) -> Dict[str, Any]:
    dataset_fe_path = score_feature_engineering_step(dataset_path=dataset_path)

    summary = predict_latest_step(
        dataset_fe_path=dataset_fe_path,
        models_dir=models_dir,
        out_path=out_path,
        mode=mode,
    )
    return summary
