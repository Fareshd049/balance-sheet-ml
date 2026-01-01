from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd

try:
    import orjson as _json
    _ORJSON = True
except Exception:
    import json as _json  # type: ignore
    _ORJSON = False

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class IngestionConfig:
    input_dir: Path                 # folder containing many *.json (companyfacts extracted)
    output_dir: Path                # dataset folder, e.g. data/processed/companyfacts_long
    max_files: int = 0              # 0 = all files
    chunk_rows: int = 250_000        # rows per parquet chunk written
    include_taxonomies: Optional[List[str]] = None  # e.g. ["us-gaap"] or None for all
    forms_keep: Optional[List[str]] = None          # e.g. ["10-K", "10-Q"] or None for all


LONG_COLUMNS = [
    "cik",
    "entity",
    "taxonomy",
    "concept",
    "unit",
    "val",
    "end",
    "fy",
    "fp",
    "form",
    "filed",
    "accn",
    "frame",
]


def _read_json(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    if _ORJSON:
        return _json.loads(data)  # type: ignore[attr-defined]
    return _json.loads(data.decode("utf-8"))  # type: ignore[no-any-return]


def iter_companyfacts_rows(obj: Dict[str, Any],
                           include_taxonomies: Optional[List[str]] = None,
                           forms_keep: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
    """
    Flatten one companyfacts JSON object into long rows.
    """
    cik_raw = obj.get("cik", "")
    cik = str(cik_raw).zfill(10) if cik_raw is not None else ""
    entity = obj.get("entityName")

    facts = obj.get("facts", {})
    if not isinstance(facts, dict):
        return

    for taxonomy, tax_data in facts.items():
        if include_taxonomies and taxonomy not in include_taxonomies:
            continue
        if not isinstance(tax_data, dict):
            continue

        for concept, concept_data in tax_data.items():
            if not isinstance(concept_data, dict):
                continue

            units = concept_data.get("units", {})
            if not isinstance(units, dict):
                continue

            for unit, points in units.items():
                if not isinstance(points, list):
                    continue

                for p in points:
                    if not isinstance(p, dict):
                        continue

                    form = p.get("form")
                    if forms_keep and form not in forms_keep:
                        continue

                    yield {
                        "cik": cik,
                        "entity": entity,
                        "taxonomy": taxonomy,
                        "concept": concept,
                        "unit": unit,
                        "val": p.get("val"),
                        "end": p.get("end"),
                        "fy": p.get("fy"),
                        "fp": p.get("fp"),
                        "form": form,
                        "filed": p.get("filed"),
                        "accn": p.get("accn"),
                        "frame": p.get("frame"),
                    }


def _ensure_dirs(cfg: IngestionConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def _parquet_writer_for_partition(output_dir: Path, cik: str, schema: pa.Schema) -> pq.ParquetWriter:
    """
    Writes each company to its own folder: output_dir/cik=XXXXXXXXXX/part-000.parquet
    """
    part_dir = output_dir / f"cik={cik}"
    part_dir.mkdir(parents=True, exist_ok=True)

    # create unique filename to avoid overwrite if rerun partially
    # (simple increment strategy)
    existing = sorted(part_dir.glob("part-*.parquet"))
    next_idx = len(existing)
    out_file = part_dir / f"part-{next_idx:03d}.parquet"

    return pq.ParquetWriter(out_file, schema=schema, compression="snappy")


def ingest_companyfacts_to_partitioned_parquet(cfg: IngestionConfig) -> Dict[str, Any]:
    """
    Main ingestion entrypoint:
    - scans json files in cfg.input_dir
    - parses each company JSON
    - writes long-format parquet partitioned by cik

    Returns basic stats for logging.
    """
    _ensure_dirs(cfg)

    files = sorted(cfg.input_dir.glob("*.json"))
    if cfg.max_files and cfg.max_files > 0:
        files = files[: cfg.max_files]

    # Define a stable Arrow schema (keeps dataset consistent)
    schema = pa.schema([
        ("cik", pa.string()),
        ("entity", pa.string()),
        ("taxonomy", pa.string()),
        ("concept", pa.string()),
        ("unit", pa.string()),
        ("val", pa.float64()),     # numeric; non-numeric will become null after coercion
        ("end", pa.string()),      # keep as string here; cast later in cleaning step
        ("fy", pa.int32()),
        ("fp", pa.string()),
        ("form", pa.string()),
        ("filed", pa.string()),
        ("accn", pa.string()),
        ("frame", pa.string()),
    ])

    n_files = 0
    n_rows = 0
    n_companies_written = 0

    for fp in files:
        obj = _read_json(fp)

        # collect rows for this company (still bounded)
        rows = list(iter_companyfacts_rows(
            obj,
            include_taxonomies=cfg.include_taxonomies,
            forms_keep=cfg.forms_keep
        ))

        n_files += 1
        if not rows:
            continue

        cik = rows[0].get("cik", "")
        if not cik:
            continue

        # Write in chunks to keep memory stable
        writer = _parquet_writer_for_partition(cfg.output_dir, cik, schema=schema)
        try:
            start = 0
            while start < len(rows):
                chunk = rows[start:start + cfg.chunk_rows]
                df = pd.DataFrame(chunk, columns=LONG_COLUMNS)

                # Coerce types gently
                df["val"] = pd.to_numeric(df["val"], errors="coerce")
                df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")

                # Convert to Arrow table w/ schema
                table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                writer.write_table(table)

                n_rows += len(chunk)
                start += cfg.chunk_rows

            n_companies_written += 1
        finally:
            writer.close()

    return {
        "files_scanned": n_files,
        "rows_written": n_rows,
        "companies_written": n_companies_written,
        "output_dir": str(cfg.output_dir),
    }
