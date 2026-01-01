import duckdb

con = duckdb.connect()

print("TOTAL ROWS:")
print(
    con.execute(
        "SELECT COUNT(*) "
        "FROM read_parquet('data/processed/companyfacts_long/cik=*/part-*.parquet')"
    ).fetchone()
)

print("\nTOP UNITS:")
print(
    con.execute(
        "SELECT unit, COUNT(*) AS c "
        "FROM read_parquet('data/processed/companyfacts_long/cik=*/part-*.parquet') "
        "GROUP BY unit ORDER BY c DESC LIMIT 10"
    ).fetchall()
)

print("\nTOP FORMS:")
print(
    con.execute(
        "SELECT form, COUNT(*) AS c "
        "FROM read_parquet('data/processed/companyfacts_long/cik=*/part-*.parquet') "
        "GROUP BY form ORDER BY c DESC LIMIT 10"
    ).fetchall()
)

con.close()
