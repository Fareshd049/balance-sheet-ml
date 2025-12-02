import pandas as pd
from pathlib import Path

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Enforce dtypes
    df["cik"] = df["cik"].astype("string")
    df["entity"] = df["entity"].astype("string")
    df["concept"] = df["concept"].astype("string")
    df["unit"] = df["unit"].astype("string").str.lower()
    df["fp"] = df["fp"].astype("string")
    df["form"] = df["form"].astype("string")
    df["accn"] = df["accn"].astype("string")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Drop rows missing critical fields
    df = df.dropna(subset=["value", "fy", "fp", "form", "end"])

    # Normalize units
    df.loc[df["unit"].str.contains("thousand", na=False), "value"] *= 1000
    df.loc[df["unit"].str.contains("million", na=False), "value"] *= 1_000_000

    # Filter by form: keep 10-Q for quarters, 10-K for FY
    mask_quarters = df["fp"].isin(["Q1","Q2","Q3"])
    mask_fy = df["fp"] == "FY"
    df = df[(mask_quarters & (df["form"] == "10-Q")) | (mask_fy & (df["form"] == "10-K"))]

    # Deduplicate: keep latest filing per cik/concept/fy/fp
    df = df.sort_values(["cik","concept","fy","fp","end"])
    df = df.drop_duplicates(subset=["cik","concept","fy","fp"], keep="last")


    return df.reset_index(drop=True)


if __name__ == "__main__":
    interim = Path("C:\\Users\\PC\\balance-sheet-ml\\data\\interim")
    in_file = interim / "clean_part0.parquet"
    out_file = interim / "cleaned_0.parquet"

    df = pd.read_parquet(in_file)
    df_clean = clean_dataframe(df)
    df_clean.to_parquet(out_file, index=False)
    print("📂 Reading:", in_file)
    print("Rows before cleaning:", len(df))
    print("Rows after cleaning:", len(df_clean))
    print("✅ Saved to:", out_file)


    print(f"✅ Cleaned dataset has {len(df_clean)} rows → {out_file}")
    print("Unique CIKs:", df_clean["cik"].nunique())
    print("Unique concepts:", df_clean["concept"].nunique())
    print("Fiscal years:", df_clean["fy"].min(), "→", df_clean["fy"].max())
