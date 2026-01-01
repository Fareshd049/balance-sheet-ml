

from typing import Any, Dict, Optional, List


from zenml import pipeline

from steps.extract_step import extract_companyfacts_step
from steps.ingestion_step import ingestion_step
from steps.quality_checks_step import quality_checks_step



@pipeline
def data_pipeline(
    zip_path: str,
    extract_dir: str,
    output_dir: str,
    max_files: int = 0,
    chunk_rows: int = 250_000,
    include_taxonomies: Optional[List[str]] = None,
    forms_keep: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    End-to-end data pipeline (Milestone A):

    companyfacts.zip
      -> extract to folder of JSONs
      -> parse & write long partitioned parquet dataset (by cik)
    """
    _ = extract_companyfacts_step(zip_path=zip_path, extract_dir=extract_dir)

    stats = ingestion_step(
        input_dir=extract_dir,
        output_dir=output_dir,
        max_files=max_files,
        chunk_rows=chunk_rows,
        include_taxonomies=include_taxonomies,
        forms_keep=forms_keep,
    )

    qc = quality_checks_step(dataset_dir=output_dir)

    return {"ingestion": stats, "quality": qc}
