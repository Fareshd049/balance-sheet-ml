from typing import Dict, Any, Tuple
from typing_extensions import Annotated
import numpy as np
import pandas as pd
from zenml import step


@step(enable_cache=False)
def split_step(
    dataset_path: str,
    target_col: str = "y_Assets",
    n_test: int = 2,
    n_val: int = 1,
) -> Tuple[
    Annotated[str, "train_path"],
    Annotated[str, "val_path"],
    Annotated[str, "test_path"],
    Annotated[Dict[str, Any], "split_stats"],
]:
    df = pd.read_parquet(dataset_path)

    # Debug prints (optional)
    print("[split_step] dataset_path:", dataset_path)
    print("[split_step] columns sample:", list(df.columns)[:40])
    print("[split_step] has cik?", "cik" in df.columns)

    # Basic validation
    if "cik" not in df.columns:
        # handle the case where cik is index
        if df.index.name == "cik":
            df = df.reset_index()
        elif isinstance(df.index, pd.MultiIndex) and "cik" in df.index.names:
            df = df.reset_index(level="cik")
        else:
            raise KeyError(f"'cik' not found. cols={list(df.columns)[:40]}")

    if "end" not in df.columns:
        raise KeyError("'end' column not found (needed for chronological split).")

    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df = df.dropna(subset=["cik", "end"]).copy()

    # Sort per company by time
    df = df.sort_values(["cik", "end"]).reset_index(drop=True)

    # Vectorized split assignment (no apply)
    g = df.groupby("cik", sort=False)
    pos = g.cumcount()                  # 0..n-1 within cik
    size = g["cik"].transform("size")   # n repeated
    from_end = (size - 1) - pos         # 0 last row, 1 previous, ...

    df["split"] = np.where(
        from_end < n_test,
        "test",
        np.where(from_end < (n_test + n_val), "val", "train"),
    )

    out_train = "./data/interim/train.parquet"
    out_val = "./data/interim/val.parquet"
    out_test = "./data/interim/test.parquet"

    df[df["split"] == "train"].drop(columns=["split"]).to_parquet(out_train, index=False)
    df[df["split"] == "val"].drop(columns=["split"]).to_parquet(out_val, index=False)
    df[df["split"] == "test"].drop(columns=["split"]).to_parquet(out_test, index=False)

    print(df["split"].value_counts(dropna=False))
    print("[split_step] companies:", df["cik"].nunique())
    print("[split_step] rows:", len(df))

    stats = {
        "train_path": out_train,
        "val_path": out_val,
        "test_path": out_test,
        "rows_train": int((df["split"] == "train").sum()),
        "rows_val": int((df["split"] == "val").sum()),
        "rows_test": int((df["split"] == "test").sum()),
        "companies": int(df["cik"].nunique()),
    }

    return out_train, out_val, out_test, stats
