

from typing import Tuple
import pandas as pd
from zenml import step


@step
def split_step(
    dataset_path: str,
    target_col: str = "y_Assets",
    n_test: int = 2,
    n_val: int = 1,
) -> Tuple[str, str, str, dict]:
    """
    Time-aware split per company.
    Returns (train_path, val_path, test_path, split_stats).
    """
    df = pd.read_parquet(dataset_path)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df = df.dropna(subset=["cik", "end", target_col]).copy()
    df = df.sort_values(["cik", "end"]).reset_index(drop=True)

    def assign_split(g: pd.DataFrame) -> pd.DataFrame:
        m = len(g)
        split = pd.Series(["train"] * m, index=g.index)
        if m >= (n_test + n_val + 1):
            split.iloc[-n_test:] = "test"
            split.iloc[-(n_test + n_val):-n_test] = "val"
        elif m >= (n_test + 1):
            split.iloc[-n_test:] = "test"
        g = g.copy()
        g["split"] = split.values
        return g

    df = df.groupby("cik", group_keys=False).apply(assign_split)

    out_train = "data/interim/train.parquet"
    out_val = "data/interim/val.parquet"
    out_test = "data/interim/test.parquet"

    df[df["split"] == "train"].drop(columns=["split"]).to_parquet(out_train, index=False)
    df[df["split"] == "val"].drop(columns=["split"]).to_parquet(out_val, index=False)
    df[df["split"] == "test"].drop(columns=["split"]).to_parquet(out_test, index=False)

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
