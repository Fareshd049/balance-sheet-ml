import duckdb

con = duckdb.connect()

# Basic per-company counts
res = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT cik) AS n_companies,
        MIN(cnt) AS min_rows_per_company,
        MAX(cnt) AS max_rows_per_company,
        AVG(cnt) AS avg_rows_per_company,
        approx_quantile(cnt, [0.25, 0.5, 0.75, 0.9]) AS quantiles
    FROM (
        SELECT cik, COUNT(*) AS cnt
        FROM read_parquet('data/interim/balance_sheet_wide.parquet')
        GROUP BY cik
    )
""").fetchall()

print("GLOBAL STATS:")
print(res)

print("\nROWS PER COMPANY (distribution):")
print(
    con.execute("""
        SELECT cnt, COUNT(*) AS n_companies
        FROM (
            SELECT cik, COUNT(*) AS cnt
            FROM read_parquet('data/interim/balance_sheet_wide.parquet')
            GROUP BY cik
        )
        GROUP BY cnt
        ORDER BY cnt
        LIMIT 20
    """).fetchall()
)

print("\nTOP COMPANIES BY HISTORY LENGTH:")
print(
    con.execute("""
        SELECT cik, COUNT(*) AS cnt
        FROM read_parquet('data/interim/balance_sheet_wide.parquet')
        GROUP BY cik
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
)

con.close()
