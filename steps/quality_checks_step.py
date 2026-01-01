

from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pyarrow.parquet as pq
from zenml import step


@step
def quality_checks_step(
    dataset_dir: str,
    out_csv: str = "data/interim/quality_report.csv",
    sample_partitions: int = 200,
) -> Dict[str, Any]:
    """
    Fast quality checks for partitioned parquet dataset:
    - counts partitions & files
    - estimates total rows from parquet metadata (no full read)
    - reads a sample to compute: date range, missing rates, top concepts, forms
    """
    base = Path(dataset_dir)
    if not base.exists():
        raise FileNotFoundError(f"Dataset dir not found: {base}")

    parquet_files = sorted(base.glob("cik=*/part-*.parquet"))
    n_files = len(parquet_files)
    n_partitions = len(list(base.glob("cik=*")))

    # total rows from parquet metadata (fast)
    total_rows_est = 0
    for f in parquet_files:
        pf = pq.ParquetFile(f)
        total_rows_est += pf.metadata.num_rows

    # sample read (bounded)
    sample_files = parquet_files[: min(sample_partitions, n_files)]
    if sample_files:
        df = pd.concat((pd.read_parquet(f) for f in sample_files), ignore_index=True)
    else:
        df = pd.DataFrame()

    # compute quick metrics from sample
    metrics: Dict[str, Any] = {
        "dataset_dir": str(base),
        "parquet_files": n_files,
        "cik_partitions": n_partitions,
        "total_rows_est": int(total_rows_est),
        "sample_files_read": len(sample_files),
        "sample_rows": int(len(df)),
    }

    if not df.empty:
        # parse end dates if present (stored as string)
        if "end" in df.columns:
            df["end_dt"] = pd.to_datetime(df["end"], errors="coerce")
            metrics["end_min_sample"] = str(df["end_dt"].min())
            metrics["end_max_sample"] = str(df["end_dt"].max())
            metrics["end_missing_rate_sample"] = float(df["end_dt"].isna().mean())

        if "val" in df.columns:
            metrics["val_missing_rate_sample"] = float(pd.isna(df["val"]).mean())

        # top concepts
        if "concept" in df.columns:
            top_concepts = df["concept"].value_counts().head(30)
            metrics["top_concepts_sample"] = top_concepts.to_dict()

        # form distribution (if present)
        if "form" in df.columns:
            metrics["forms_sample"] = df["form"].value_counts().head(20).to_dict()

        # units distribution
        if "unit" in df.columns:
            metrics["units_sample"] = df["unit"].value_counts().head(20).to_dict()

        # taxonomy distribution
        if "taxonomy" in df.columns:
            metrics["taxonomy_sample"] = df["taxonomy"].value_counts().to_dict()

    # write a small flat report CSV (human-friendly)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # flatten dict for CSV (nested dicts become strings)
    flat = {k: (v if not isinstance(v, dict) else str(v)) for k, v in metrics.items()}
    pd.DataFrame([flat]).to_csv(out_path, index=False)

    metrics["report_csv"] = str(out_path)
    return metrics
