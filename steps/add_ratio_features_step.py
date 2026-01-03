

from typing import Dict, Any, Annotated
import numpy as np
import pandas as pd
from zenml import step


def _safe_div(num: pd.Series, den: pd.Series, eps: float = 1e-9) -> pd.Series:
    den_ok = den.notna() & (den.abs() > eps)
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    out.loc[den_ok] = (num.loc[den_ok] / den.loc[den_ok]).astype("float64")
    return out


@step
def add_ratio_features_step(
    dataset_path: str,
    out_path: str = "data/processed/balance_sheet_model_with_ratios.parquet",
    clip_abs: float = 20.0,   # clips extreme ratios
) -> tuple[
    Annotated[str, "dataset_out_path"],
    Annotated[Dict[str, Any], "ratio_stats"],
]:
    df = pd.read_parquet(dataset_path)

    # Ensure key cols
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df = df.dropna(subset=["cik", "end"]).copy()

    # Helpers for missing columns
    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    Assets = col("Assets")
    Liab = col("Liabilities")
    Equity = col("StockholdersEquity")

    Cash = col("CashAndCashEquivalentsAtCarryingValue")
    AssetsCur = col("AssetsCurrent")
    LiabCur = col("LiabilitiesCurrent")
    PPE = col("PropertyPlantAndEquipmentNet")
    Goodwill = col("Goodwill")
    Intang = col("IntangibleAssetsNetExcludingGoodwill")

    DebtCur = col("DebtCurrent")
    LTD = col("LongTermDebt")
    LTDcur = col("LongTermDebtCurrent")

    DebtTotal = DebtCur.fillna(0) + LTD.fillna(0) + LTDcur.fillna(0)

    # --- Ratios ---
    df["cash_to_assets"] = _safe_div(Cash, Assets)
    df["current_assets_to_assets"] = _safe_div(AssetsCur, Assets)
    df["liab_to_assets"] = _safe_div(Liab, Assets)
    df["equity_to_assets"] = _safe_div(Equity, Assets)

    df["current_liab_to_liab"] = _safe_div(LiabCur, Liab)
    df["ppe_to_assets"] = _safe_div(PPE, Assets)
    df["goodwill_to_assets"] = _safe_div(Goodwill, Assets)
    df["intangibles_to_assets"] = _safe_div(Intang, Assets)

    wc = AssetsCur - LiabCur
    df["wc_to_assets"] = _safe_div(wc, Assets)

    df["debt_to_assets"] = _safe_div(DebtTotal, Assets)

    # debt_to_equity only meaningful if equity > 0 (avoid sign weirdness)
    equity_pos = Equity.where(Equity > 0)
    df["debt_to_equity"] = _safe_div(DebtTotal, equity_pos)

    # Clip to stabilize (optional but recommended)
    ratio_cols = [
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
    for c in ratio_cols:
        df[c] = df[c].clip(lower=-clip_abs, upper=clip_abs)

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    stats = {
        "in_path": dataset_path,
        "out_path": out_path,
        "rows": int(len(df)),
        "ratio_cols": ratio_cols,
        "missing_rate": {c: float(df[c].isna().mean()) for c in ratio_cols},
    }
    return out_path, stats
