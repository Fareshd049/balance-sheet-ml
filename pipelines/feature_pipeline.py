# pipelines/feature_pipeline.py



from typing import Dict, Any, Optional, List
from zenml import pipeline

from steps.build_balance_sheet_wide_step import build_balance_sheet_wide_step


@pipeline
def feature_pipeline(
    long_dataset_dir: str,
    out_path: str,
    unit_keep: str = "USD",
    forms_keep: Optional[List[str]] = None,
    min_end: str = "1980-01-01",
    max_end: Optional[str] = None,
) -> Dict[str, Any]:
    stats = build_balance_sheet_wide_step(
        long_dataset_dir=long_dataset_dir,
        out_path=out_path,
        unit_keep=unit_keep,
        forms_keep=forms_keep,
        min_end=min_end,
        max_end=max_end,
    )
    return stats
