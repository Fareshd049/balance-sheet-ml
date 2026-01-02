

from typing import Dict, Any, Optional, List
from zenml import pipeline

from steps.build_model_base_step import build_model_base_step


@pipeline
def model_base_pipeline(
    wide_path: str,
    out_path: str,
    concepts: Optional[List[str]] = None,
    min_obs_per_company: int = 4,
) -> Dict[str, Any]:
    stats = build_model_base_step(
        wide_path=wide_path,
        out_path=out_path,
        concepts=concepts,
        min_obs_per_company=min_obs_per_company,
    )
    return stats
