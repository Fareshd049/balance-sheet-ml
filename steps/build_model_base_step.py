

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from zenml import step


def _safe_logdiff(curr: pd.Series, prev: pd.Series) -> pd.Series:
    """log(curr) - log(prev) but only where both > 0, else NaN."""
    mask = (curr > 0) & (prev > 0) & curr.notna() & prev.notna()
    out = pd.Series(np.nan, index=curr.index, dtype="float64")
    out.loc[mask] = np.log(curr.loc[mask]) - np.log(prev.loc[mask])
    return out


@step
def build_model_base_step(
    wide_path: str,
    out_path: str = "data/interim/balance_sheet_model_base.parquet",
    concepts: Optional[List[str]] = None,
    min_obs_per_company: int = 5,
    max_abs_logdiff: float = 3.0,  # ~ e^3 ≈ 20x change; clips crazy outliers
) -> Dict[str, Any]:
    """
    Build ML-ready dataset from balance_sheet_wide:
    - sort by (cik, end)
    - keep companies with enough history
    - basic accounting reconciliation for A/L/E
    - create lag-1 values
    - create targets as safe log-differences (fallback to raw diff when logdiff NaN)
    """
    df = pd.read_parquet(wide_path)

    # Ensure types
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df = df.dropna(subset=["cik", "end"]).copy()

    # If user didn't pass concepts, infer from columns (excluding id columns)
    id_cols = {"cik", "end"}
    if concepts is None:
        concepts = [c for c in df.columns if c not in id_cols]

    # Keep only the expected columns (safety)
    keep_cols = ["cik", "end"] + [c for c in concepts if c in df.columns]
    df = df[keep_cols].copy()

    # Sort time-series
    df = df.sort_values(["cik", "end"]).reset_index(drop=True)

    # Drop exact duplicates (cik,end) if they exist, keep last row
    # (should be rare after your dedupe, but safe)
    df = df.drop_duplicates(subset=["cik", "end"], keep="last")

    # Filter companies with enough observations
    counts = df.groupby("cik").size()
    keep_ciks = counts[counts >= min_obs_per_company].index
    df = df[df["cik"].isin(keep_ciks)].copy()

    # --- Basic reconciliation for core identity: Assets = Liabilities + Equity ---
    # We'll only impute if two exist and one missing.
    if "Assets" in df.columns and "Liabilities" in df.columns and "StockholdersEquity" in df.columns:
        A = df["Assets"]
        L = df["Liabilities"]
        E = df["StockholdersEquity"]

        # Fill missing L = A - E
        df.loc[L.isna() & A.notna() & E.notna(), "Liabilities"] = (A - E)

        # Fill missing E = A - L
        df.loc[E.isna() & A.notna() & L.notna(), "StockholdersEquity"] = (A - L)

        # Fill missing A = L + E
        df.loc[A.isna() & L.notna() & E.notna(), "Assets"] = (L + E)

        # Optional: if LiabilitiesAndStockholdersEquity exists, use it as check (no overwrite)
        # We keep it untouched for now.
    
    # --- Eligibility filter: require enough balance-sheet info per row ---
    concept_cols = [c for c in concepts if c in df.columns]

    # Keep rows where at least 3 concepts are present (non-null)
    required_nonnull = 3
    df = df[df[concept_cols].notna().sum(axis=1) >= required_nonnull].copy()

    # Also require Assets exists
    if "Assets" in df.columns:
        df = df[df["Assets"].notna()].copy()
    
    # Require lag for Assets so y_Assets is defined (trainable rows only)
    if "Assets_lag1" in df.columns:
        df = df[df["Assets_lag1"].notna()].copy()

    # --- Build lag features and targets ---
    # Lag-1 for each concept
    for c in concepts:
        if c in df.columns:
            df[f"{c}_lag1"] = df.groupby("cik")[c].shift(1)

    # Targets: safe logdiff where possible; otherwise raw diff
    # y_{c} = logdiff if valid else diff
    for c in concepts:
        if c not in df.columns:
            continue
        curr = df[c]
        prev = df[f"{c}_lag1"]
        logd = _safe_logdiff(curr, prev)
        diff = curr - prev

        # use logdiff where available, else diff
        y = logd.copy()
        y = y.where(~y.isna(), diff)

        # Clip extreme changes to stabilize training
        # (logdiff clipped; diff also clipped by quantiles below)
        y = y.clip(lower=-max_abs_logdiff, upper=max_abs_logdiff)

        df[f"y_{c}"] = y
        df[f"y_{c}_is_log"] = (~logd.isna()).astype("int8")

    # Drop rows where lag1 is missing for ALL key concepts (first obs per company)
    lag_cols = [f"{c}_lag1" for c in concepts if f"{c}_lag1" in df.columns]
    df = df.dropna(subset=lag_cols, how="all").copy()

    # Save
    out_dir = pd.io.common.get_handle(out_path, "wb").handle.name  # ensures path exists check handled by OS
    # Create directory
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df.to_parquet(out_path, index=False)

    return {
        "wide_path": wide_path,
        "out_path": out_path,
        "rows": int(len(df)),
        "companies": int(df["cik"].nunique()),
        "min_obs_per_company": int(min_obs_per_company),
        "concepts": len(concepts),
    }
