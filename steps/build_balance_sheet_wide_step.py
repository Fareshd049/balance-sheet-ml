# steps/build_balance_sheet_wide_step.py


from typing import Dict, Any, List, Optional
from datetime import date

from zenml import step
import duckdb

from src.features.balance_sheet_concepts import BALANCE_SHEET_CONCEPTS


@step
def build_balance_sheet_wide_step(
    long_dataset_dir: str,
    out_path: str = "data/interim/balance_sheet_wide.parquet",
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

        forms_sql = ",".join([f"'{f}'" for f in forms_keep])
    concepts_sql = ",".join([f"'{c}'" for c in concepts])

    sql = f"""
    COPY (
        WITH base AS (
            SELECT
                cik,
                try_cast("end" AS DATE) AS end_dt,
                concept,
                val,
                form,
                try_cast(filed AS DATE) AS filed_dt,
                accn
            FROM read_parquet('{long_dataset_dir}/cik=*/part-*.parquet')
            WHERE taxonomy = 'us-gaap'
              AND unit = '{unit_keep}'
              AND form IN ({forms_sql})
              AND concept IN ({concepts_sql})
              AND try_cast("end" AS DATE) IS NOT NULL
              AND try_cast("end" AS DATE) BETWEEN DATE '{min_end}' AND DATE '{max_end}'
        ),
        best_key AS (
            -- "Best filing per (cik, end)" using aggregate keys (no window)
            SELECT
                cik,
                end_dt,
                MAX(filed_dt) AS best_filed
            FROM base
            GROUP BY cik, end_dt
        ),
        base2 AS (
            -- keep only rows on the best filed date for that (cik, end)
            SELECT b.*
            FROM base b
            JOIN best_key k
              ON b.cik = k.cik
             AND b.end_dt = k.end_dt
             AND b.filed_dt = k.best_filed
        ),
        best_key2 AS (
            -- if multiple filings share the same filed date, prefer amended forms then latest accn
            SELECT
                cik,
                end_dt,
                MAX(CASE WHEN form LIKE '%/A' THEN 1 ELSE 0 END) AS prefer_amend
            FROM base2
            GROUP BY cik, end_dt
        ),
        base3 AS (
            SELECT b.*
            FROM base2 b
            JOIN best_key2 k
              ON b.cik = k.cik
             AND b.end_dt = k.end_dt
            WHERE (k.prefer_amend = 0)
               OR (k.prefer_amend = 1 AND b.form LIKE '%/A')
        ),
        best_accn AS (
            SELECT
                cik,
                end_dt,
                MAX(accn) AS best_accn
            FROM base3
            GROUP BY cik, end_dt
        ),
        chosen AS (
            SELECT b.cik, b.end_dt, b.concept, b.val
            FROM base3 b
            JOIN best_accn a
              ON b.cik = a.cik
             AND b.end_dt = a.end_dt
             AND b.accn = a.best_accn
        )
        SELECT
            cik,
            end_dt AS "end",
            {", ".join(pivot_exprs)}
        FROM chosen
        GROUP BY cik, "end"
    )
    TO '{out_path}' (FORMAT 'parquet', COMPRESSION 'snappy');
    """


    con = duckdb.connect(database=":memory:")
    # DuckDB tuning for low-RAM machines
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET threads=2;")                 # reduce peak memory
    con.execute("SET memory_limit='10GB';")       # keep under your 12.5GB
    con.execute("SET temp_directory='data/interim/duckdb_tmp';")
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
