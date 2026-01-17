from typing import Dict, Any, Optional
from pathlib import Path
from zenml import pipeline

from steps.score_feature_engineering_step import score_feature_engineering_step
from steps.predict_latest_step import predict_latest_step
from steps.parse_company_facts_json_step import parse_company_facts_json_step
from steps.postprocess_scores_insights_step import postprocess_scores_insights_step


@pipeline(enable_cache=True)
def inference_pipeline(
    dataset_path: Optional[str],
    json_path: Optional[str],
    models_dir: str,
    out_path: str,
    out_scored_path: str,
    mode: str = "latest_per_company",
    delta_mode: str = "logdiff",
) -> Dict[str, Any]:

    if mode == "single_company_json":
        if not json_path:
            raise ValueError("json_path is required when mode='single_company_json'")

        stem = Path(json_path).stem
        raw_out = f"data/processed/single_raw_{stem}.parquet"

        raw_path = parse_company_facts_json_step(
            json_path=json_path,
            out_parquet_path=raw_out,
        )

        dataset_fe_path = score_feature_engineering_step(dataset_path=raw_path)

        preds_summary = predict_latest_step(
            dataset_fe_path=dataset_fe_path,
            models_dir=models_dir,
            out_path=out_path,
            mode="all_rows",
            min_end=None,
            year_filter=None,
        )

        final_summary = postprocess_scores_insights_step(
            dataset_fe_path=dataset_fe_path,
            preds_path=out_path,
            out_path=out_scored_path,
            delta_mode=delta_mode,
            preds_summary=preds_summary,  # enforce order
        )

        return {"predict": preds_summary, "final": final_summary}

    if not dataset_path:
        raise ValueError("dataset_path is required when mode='latest_per_company'")

    dataset_fe_path = score_feature_engineering_step(dataset_path=dataset_path)

    preds_summary = predict_latest_step(
        dataset_fe_path=dataset_fe_path,
        models_dir=models_dir,
        out_path=out_path,
        mode=mode,
        min_end="2025-01-01",
        year_filter=None,
    )

    final_summary = postprocess_scores_insights_step(
        dataset_fe_path=dataset_fe_path,
        preds_path=out_path,
        out_path=out_scored_path,
        delta_mode=delta_mode,
        preds_summary=preds_summary,
    )

    return {"predict": preds_summary, "final": final_summary}
