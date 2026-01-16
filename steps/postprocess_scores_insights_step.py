# steps/postprocess_scores_insights_step.py
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import numpy as np
import pandas as pd
from zenml import step

EPS = 1e-9

BASE_COLS = {
    "Assets": "y_Assets",
    "Liabilities": "y_Liabilities",
    "StockholdersEquity": "y_StockholdersEquity",
    "AssetsCurrent": "y_AssetsCurrent",
    "LiabilitiesCurrent": "y_LiabilitiesCurrent",
    "CashAndCashEquivalentsAtCarryingValue":
        "y_CashAndCashEquivalentsAtCarryingValue",
}

def _safe_div(a, b):
    return a / (b + EPS)

def _to_next_value(current: float, pred: float, delta_mode: str) -> float:
    """
    delta_mode:
      - 'logdiff': pred is delta in log1p-space => next = expm1(log1p(curr) + pred)
      - 'pct':     pred is percent change => next = curr * (1 + pred)
      - 'abs':     pred is absolute delta => next = curr + pred
    """
    if current is None or np.isnan(current):
        return np.nan
    if pred is None or np.isnan(pred):
        return np.nan

    if delta_mode == "logdiff":
        return float(np.expm1(np.log1p(max(current, 0.0)) + pred))
    if delta_mode == "pct":
        return float(current * (1.0 + pred))
    if delta_mode == "abs":
        return float(current + pred)

    raise ValueError(f"Unknown delta_mode={delta_mode}")

def _compute_ratios(row: pd.Series, prefix: str = "") -> Dict[str, float]:
    A  = row.get(prefix + "Assets", np.nan)
    L  = row.get(prefix + "Liabilities", np.nan)
    E  = row.get(prefix + "StockholdersEquity", np.nan)
    CA = row.get(prefix + "AssetsCurrent", np.nan)
    CL = row.get(prefix + "LiabilitiesCurrent", np.nan)
    C  = row.get(prefix + "CashAndCashEquivalentsAtCarryingValue", np.nan)

    out = {}
    out[prefix + "current_ratio"] = _safe_div(CA, CL)
    out[prefix + "cash_ratio"]    = _safe_div(C, CL)
    out[prefix + "debt_to_assets"]= _safe_div(L, A)
    out[prefix + "equity_to_assets"]= _safe_div(E, A)
    out[prefix + "debt_to_equity"]= _safe_div(L, E)
    out[prefix + "wc_to_assets"]  = _safe_div(CA - CL, A)
    return out

def _score_solvency(r_now: Dict[str, float], r_next: Dict[str, float]) -> float:
    # Simple, transparent scoring (0..100)
    # Higher equity/assets and lower debt/assets is better.
    ea = np.clip(r_next.get("next_equity_to_assets", np.nan), -1, 1)
    da = np.clip(r_next.get("next_debt_to_assets", np.nan), 0, 5)

    # map to 0..100 (very simple)
    s1 = 50.0 + 50.0 * np.nan_to_num(ea, nan=0.0)          # equity/assets
    s2 = 100.0 / (1.0 + np.nan_to_num(da, nan=1.0))        # debt/assets penalty
    return float(np.clip(0.6 * s1 + 0.4 * s2, 0, 100))

def _score_liquidity(r_now: Dict[str, float], r_next: Dict[str, float]) -> float:
    cr = np.clip(r_next.get("next_current_ratio", np.nan), 0, 5)
    cashr = np.clip(r_next.get("next_cash_ratio", np.nan), 0, 5)
    wc = np.clip(r_next.get("next_wc_to_assets", np.nan), -1, 1)

    s1 = 100.0 * np.clip(cr / 2.0, 0, 1)       # current ratio target ~2
    s2 = 100.0 * np.clip(cashr / 0.5, 0, 1)    # cash ratio target ~0.5
    s3 = 50.0 + 50.0 * wc
    return float(np.clip(0.45 * s1 + 0.35 * s2 + 0.20 * s3, 0, 100))

def _insights(row: pd.Series) -> List[Dict[str, Any]]:
    insights = []

    # Accounting identity check
    A = row.get("Assets", np.nan)
    L = row.get("Liabilities", np.nan)
    E = row.get("StockholdersEquity", np.nan)
    if np.isfinite(A) and np.isfinite(L) and np.isfinite(E):
        gap = abs(A - (L + E))
        if gap > max(1e-6, 0.02 * abs(A)):  # 2% tolerance
            insights.append({
                "type": "warning",
                "title": "Balance sheet does not balance",
                "detail": f"Assets != Liabilities + Equity (gap={gap:,.0f}). Check input mapping/period."
            })

    # Liquidity flags
    if row.get("next_current_ratio", np.nan) < 1.0:
        insights.append({"type":"risk","title":"Low liquidity", "detail":"Projected current ratio < 1.0 (short-term stress)."})
    if row.get("next_cash_ratio", np.nan) < 0.1:
        insights.append({"type":"risk","title":"Low cash coverage", "detail":"Projected cash ratio < 0.1 (cash may not cover current liabilities)."})

    # Solvency flags
    if row.get("next_debt_to_assets", np.nan) > 0.7:
        insights.append({"type":"risk","title":"High leverage", "detail":"Projected liabilities/assets > 0.70 (high solvency pressure)."})
    if row.get("StockholdersEquity", np.nan) < 0 or row.get("next_StockholdersEquity", np.nan) < 0:
        insights.append({"type":"risk","title":"Negative equity", "detail":"Equity is negative (or projected negative), which is a strong distress signal."})

    # Deterioration signals (next vs now)
    if row.get("next_wc_to_assets", np.nan) < row.get("wc_to_assets", np.nan) - 0.05:
        insights.append({"type":"info","title":"Working capital deteriorating", "detail":"Projected working capital / assets decreases materially."})

    return insights

@step(enable_cache=False)
def postprocess_scores_insights_step(
    dataset_fe_path: str,
    preds_path: str,
    out_path: str,
    delta_mode: str = "logdiff",
) -> Dict[str, Any]:
    df = pd.read_parquet(dataset_fe_path)
    preds = pd.read_parquet(preds_path)

    # join on cik + end (or other time col)
    time_col = "end" if "end" in preds.columns else preds.columns[1]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    preds[time_col] = pd.to_datetime(preds[time_col], errors="coerce")

    merged = preds.merge(df, on=["cik", time_col], how="left", suffixes=("", "_feat"))

    # Ensure we have base “today” columns (either SEC names or your internal names)
    # If your dataset uses different names, add mapping here once.
    for col in BASE_COLS.keys():
        if col not in merged.columns and col.replace("AndCashEquivalentsAtCarryingValue","") in merged.columns:
            merged[col] = merged[col.replace("AndCashEquivalentsAtCarryingValue","")]

    # Build next-period values from predictions
    for base_col, target_name in BASE_COLS.items():
        pred_col = f"pred_{target_name}"
        if pred_col not in merged.columns:
            raise ValueError(
                f"Missing prediction column {pred_col}. "
                f"Available columns: {list(merged.columns)}"
            )
        merged[f"next_{base_col}"] = [
            _to_next_value(cur, p, delta_mode)
            for cur, p in zip(merged.get(base_col, np.nan), merged[pred_col])
        ]

    # Ratios now/next
    now_ratios = merged.apply(lambda r: _compute_ratios(r, prefix=""), axis=1, result_type="expand")
    next_ratios = merged.apply(lambda r: _compute_ratios(r, prefix="next_"), axis=1, result_type="expand")
    merged = pd.concat([merged, now_ratios.add_prefix(""), next_ratios.add_prefix("next_")], axis=1)

    # Scores
    merged["solvency_score"] = merged.apply(
        lambda r: _score_solvency(_compute_ratios(r,""), _compute_ratios(r,"next_")),
        axis=1
    )
    merged["liquidity_score"] = merged.apply(
        lambda r: _score_liquidity(_compute_ratios(r,""), _compute_ratios(r,"next_")),
        axis=1
    )

    # Insights
    merged["insights"] = merged.apply(_insights, axis=1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    summary = {
        "rows": int(len(merged)),
        "out_path": out_path,
        "delta_mode": delta_mode,
        "columns": list(merged.columns),
    }
    with open(str(Path(out_path).with_suffix(".json")), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
