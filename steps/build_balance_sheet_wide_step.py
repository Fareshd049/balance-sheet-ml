# steps/build_balance_sheet_wide_step.py


from typing import Dict, Any, List, Optional
from datetime import date

from zenml import step
import duckdb

from src.features.balance_sheet_concepts import BALANCE_SHEET_CONCEPTS


@step
def build_balance_sheet_wide_step(
    long_dataset_dir: str,
    out_path: str = "data/processed/balance_sheet_wide.parquet",
    concepts: Optional[List[str]] = None,
    unit_keep: str = "USD",
    forms_keep: Optional[List[str]] = None,
    min_end: str = "1980-01-01",
    max_end: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds a pure Balance Sheet wide table:

    Input (long):
      columns: cik, concept, unit, val, end, form, ...
    Output (wide):
      index: (cik, end)
      columns: balance sheet concepts
    """
    concepts = concepts or BALANCE_SHEET_CONCEPTS

    # Date bounds
    if max_end is None:
        # allow up to next year to avoid dropping near-future edge cases, but remove absurd 2201, etc.
        max_end = f"{date.today().year + 1}-12-31"

    # Default forms: keep the regular financial statements.
    # You can expand later (e.g. 20-F / 40-F) if you want foreign filers included.
    if forms_keep is None:
        forms_keep = ["10-K", "10-Q", "10-K/A", "10-Q/A"]

    # Build SQL pivot via conditional aggregation (works out-of-core)
    # NOTE: end stored as string in your ingestion; cast to DATE safely.
    select_cols = ["cik", "end_dt"]
    pivot_exprs = []
    for c in concepts:
        pivot_exprs.append(
            f"max(case when concept = '{c}' then val end) as \"{c}\""
        )

    sql = f"""
    COPY (
        WITH filtered AS (
            SELECT
                cik,
                try_cast("end" as DATE) as end_dt,
                concept,
                val,
                unit,
                form
            FROM read_parquet('{long_dataset_dir}/cik=*/part-*.parquet')
            WHERE taxonomy = 'us-gaap'
              AND unit = '{unit_keep}'
              AND form IN ({",".join([f"'{f}'" for f in forms_keep])})
              AND try_cast("end" as DATE) IS NOT NULL
              AND try_cast("end" as DATE) BETWEEN DATE '{min_end}' AND DATE '{max_end}'
        )
        SELECT
            cik,
            end_dt as "end",
            {", ".join(pivot_exprs)}
        FROM filtered
        GROUP BY cik, "end"
    )
    TO '{out_path}' (FORMAT 'parquet', COMPRESSION 'snappy');
    """

    con = duckdb.connect(database=":memory:")
    con.execute(sql)
    con.close()

    return {
        "out_path": out_path,
        "concepts_used": len(concepts),
        "unit_keep": unit_keep,
        "forms_keep": forms_keep,
        "min_end": min_end,
        "max_end": max_end,
    }
