

from pathlib import Path
from typing import Any, Dict, Optional, List

from zenml import step

from src.ingestion import IngestionConfig, ingest_companyfacts_to_partitioned_parquet


@step
def ingestion_step(
    input_dir: str,
    output_dir: str,
    max_files: int = 0,
    chunk_rows: int = 250_000,
    include_taxonomies: Optional[List[str]] = None,
    forms_keep: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    ZenML step wrapper for ingestion.

    Expected:
    - input_dir points to extracted JSON folder: data/raw/companyfacts/
    Output:
    - output_dir is a partitioned parquet dataset folder: data/processed/companyfacts_long/
    """
    cfg = IngestionConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        max_files=max_files,
        chunk_rows=chunk_rows,
        include_taxonomies=include_taxonomies,
        forms_keep=forms_keep,
    )
    return ingest_companyfacts_to_partitioned_parquet(cfg)
