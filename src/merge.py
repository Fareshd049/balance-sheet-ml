import pandas as pd
import glob
from pathlib import Path

if __name__ == "__main__":
    interim = Path("data/interim")
    out_file = interim / "clean_all.parquet"

    # Find all batch parquet files
    files = glob.glob(str(interim / "clean_part*.parquet"))
    if not files:
        raise FileNotFoundError("⚠️ No clean_part parquet files found in data/interim")

    print(f"Found {len(files)} batch files, merging...")

    # Concatenate all parts
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Save merged dataset
    df.to_parquet(out_file, index=False)
    print(f"✅ Final dataset has {len(df)} rows → {out_file}")
