import numpy as np
import pandas as pd
from zenml import step

EPS = 1e-9

RATIO_COLS = [
    "cash_to_assets",
    "current_assets_to_assets",
    "liab_to_assets",
    "equity_to_assets",
    "current_liab_to_liab",
    "ppe_to_assets",
    "goodwill_to_assets",
    "intangibles_to_assets",
    "wc_to_assets",
    "debt_to_assets",
    "debt_to_equity",
]

@step(enable_cache=True)
def score_feature_engineering_step(dataset_path: str) -> str:
    df = pd.read_parquet(dataset_path)

    # choose time col
    if "end" in df.columns:
        time_col = "end"
    elif "period_end_date" in df.columns:
        time_col = "period_end_date"
    elif "date" in df.columns:
        time_col = "date"
    else:
        time_col = None

    if time_col is not None and "cik" in df.columns:
        df = df.sort_values(["cik", time_col])
    elif "cik" in df.columns:
        df = df.sort_values(["cik"])

    ratio_cols = [c for c in RATIO_COLS if c in df.columns]
    if ratio_cols and "cik" in df.columns:
        g = df.groupby("cik", group_keys=False)

        for c in ratio_cols:
            lag1 = g[c].shift(1)
            df[f"{c}_delta1"] = df[c] - lag1
            df[f"{c}_pct1"] = (df[c] - lag1) / (np.abs(lag1) + EPS)

            df[f"{c}_rollmean4"] = (
                g[c].rolling(4, min_periods=2).mean().reset_index(level=0, drop=True)
            )
            df[f"{c}_rollstd4"] = (
                g[c].rolling(4, min_periods=2).std().reset_index(level=0, drop=True)
            )

    # optional extras (usually not triggered in your dataset; harmless)
    if {"Cash", "CurrentLiabilities"}.issubset(df.columns):
        denom = df["CurrentLiabilities"].replace(0, np.nan)
        df["cash_to_current_liab"] = df["Cash"] / denom

    if {"CurrentAssets", "CurrentLiabilities"}.issubset(df.columns):
        denom = df["CurrentLiabilities"].replace(0, np.nan)
        df["current_ratio"] = df["CurrentAssets"] / denom

    if {"Debt", "Cash", "Assets"}.issubset(df.columns):
        denom = df["Assets"].replace(0, np.nan)
        df["net_debt_to_assets"] = (df["Debt"] - df["Cash"]) / denom

    df = df.replace([np.inf, -np.inf], np.nan)

    out_path = dataset_path.replace(".parquet", "_fe.parquet")
    df.to_parquet(out_path, index=False)
    return out_path
