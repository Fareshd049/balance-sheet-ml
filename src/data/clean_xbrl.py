#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_xbrl(input_file: Path, output_file: Path):
    df = pd.read_parquet(input_file)

    # Step 1: Drop rows without end or value
    df = df.dropna(subset=['end','value'])

    # Step 2: Ensure correct dtypes
    df['cik'] = df['cik'].astype(str)
    df['end'] = pd.to_datetime(df['end'], errors='coerce')
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Step 3: Handle duplicates (keep latest filing)
    df = df.sort_values(['cik','end','accn'])
    df = df.drop_duplicates(subset=['cik','end','concept'], keep='last')

    # Step 4: Normalize units
    def normalize_units(row):
        if row['unit'] == 'USD Thousands':
            return row['value'] * 1000
        elif row['unit'] == 'USD Millions':
            return row['value'] * 1_000_000
        return row['value']
    df['value'] = df.apply(normalize_units, axis=1)
    df['unit'] = 'USD'

    # Step 5: Pivot long → wide
    wide = df.pivot_table(
        index=['cik','entity','end'],   # keep entity here
        columns='concept',
        values='value',
        aggfunc='last'
    ).reset_index()

    wide.to_parquet(output_file, index=False)
    logger.info("Saved cleaned wide data to %s", output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    clean_xbrl(Path(args.input), Path(args.output))
