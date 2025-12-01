import pandas as pd
from pathlib import Path

def pivot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Pivot: concepts become columns, values are numeric
    wide = df.pivot_table(
        index=["cik","entity","fy","fp","end","accn"],  # 🔹 keep metadata
        columns="concept",
        values="value",
        aggfunc="last"   # keep last value if duplicates remain
    )

    # Reset index so keys are columns again
    wide = wide.reset_index()



    return wide


if __name__ == "__main__":
    interim = Path("data/interim")
    in_file = interim / "cleaned_0.parquet"
    out_file = Path("data/processed") / "wide_0.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(" Reading:", in_file)
    df = pd.read_parquet(in_file)
    print("Rows before pivot:", len(df))

    df_wide = pivot_dataframe(df)
    df_wide.to_parquet(out_file, index=False)

    print(" Pivoted dataset has shape", df_wide.shape, "→", out_file)
    print("Features (concepts):", len(df_wide.columns) - 6)  # exclude keys
