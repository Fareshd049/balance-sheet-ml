from typing import Dict, Any, Optional
from zenml import pipeline

from steps.score_feature_engineering_step import score_feature_engineering_step
from steps.predict_latest_step import predict_latest_step
from steps.parse_company_facts_json_step import parse_company_facts_json_step

@pipeline(enable_cache=True)
def inference_pipeline(
    # Batch mode input
    dataset_path: Optional[str],
    # Single JSON mode input
    json_path: Optional[str],
    # Common
    models_dir: str,
    out_path: str,
    mode: str = "latest_per_company",
) -> Dict[str, Any]:
    """
    mode:
      - "latest_per_company": uses dataset_path (parquet)
      - "single_company_json": uses json_path (SEC company facts json)
    """
    if mode == "single_company_json":
        if not json_path:
            raise ValueError("json_path is required when mode='single_company_json'")
        raw_path = parse_company_facts_json_step(
            json_path=json_path,
            out_parquet_path=out_path.replace(".parquet", "_single_raw.parquet"),
        )
        dataset_fe_path = score_feature_engineering_step(dataset_path=raw_path)

        summary = predict_latest_step(
            dataset_fe_path=dataset_fe_path,
            models_dir=models_dir,
            out_path=out_path,
            mode="all_rows",  # score the single row
            min_end=None,
            year_filter=None,
        )
        return summary

    # default: batch from dataset_path
    if not dataset_path:
        raise ValueError("dataset_path is required when mode='latest_per_company'")

    dataset_fe_path = score_feature_engineering_step(dataset_path=dataset_path)

    summary = predict_latest_step(
        dataset_fe_path=dataset_fe_path,
        models_dir=models_dir,
        out_path=out_path,
        mode=mode,
    )
    return summary
