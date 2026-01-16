# steps/parse_company_facts_json_step.py
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import numpy as np
from zenml import step

EPS = 1e-9

REQUIRED_TAGS = [
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "CashAndCashEquivalentsAtCarryingValue",
]

def _latest_usd(tag_block: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """Return (val, end_date) for latest USD fact in a tag block."""
    usd = tag_block.get("units", {}).get("USD", [])
    if not usd:
        return None, None
    # pick latest by 'end'
    latest = max(usd, key=lambda x: x.get("end", "0000-00-00"))
    return latest.get("val"), latest.get("end")

@step(enable_cache=True)
def parse_company_facts_json_step(
    json_path: str,
    out_parquet_path: str,
) -> str:
    """
    Parse SEC company facts JSON and output a single-row parquet
    with base fields + core ratios (so downstream steps can run).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    row: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    # Extract base balance sheet values
    ends = []
    for tag in REQUIRED_TAGS:
        if tag not in data:
            row[tag] = np.nan
            meta[f"{tag}_end"] = None
            continue
        val, end = _latest_usd(data[tag])
        row[tag] = float(val) if val is not None else np.nan
        meta[f"{tag}_end"] = end
        if end:
            ends.append(end)

    # Add cik + end for compatibility with your predict_latest_step
    row["cik"] = Path(json_path).stem.replace("CIK", "")
    row["end"] = max(ends) if ends else None

    # ---- Create ratio cols used in your project ----
    assets = row.get("Assets", np.nan)
    liab = row.get("Liabilities", np.nan)
    eq = row.get("StockholdersEquity", np.nan)
    ca = row.get("AssetsCurrent", np.nan)
    cl = row.get("LiabilitiesCurrent", np.nan)
    cash = row.get("CashAndCashEquivalentsAtCarryingValue", np.nan)

    # ratios (match names you use in FE step)
    row["cash_to_assets"] = cash / (assets + EPS)
    row["current_assets_to_assets"] = ca / (assets + EPS)
    row["liab_to_assets"] = liab / (assets + EPS)
    row["equity_to_assets"] = eq / (assets + EPS)
    row["current_liab_to_liab"] = cl / (liab + EPS)
    row["wc_to_assets"] = (ca - cl) / (assets + EPS)

    # If you don’t have PPE/Goodwill/Intangibles/Debt from JSON, keep them NaN.
    # The model + _load_xy already handles missing feature cols by creating them.
    # (HGB can handle NaNs.)
    df = pd.DataFrame([row])

    outp = Path(out_parquet_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)

    return str(outp)
