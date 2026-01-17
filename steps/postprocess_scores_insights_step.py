from typing import Dict, Any, Optional, List
from pathlib import Path
import json

import numpy as np
import pandas as pd
from zenml import step

EPS = 1e-9

# Map: base column in features -> prediction column suffix in preds parquet
BASE_COLS = {
    "Assets": "y_Assets",
    "Liabilities": "y_Liabilities",
    "StockholdersEquity": "y_StockholdersEquity",
    "AssetsCurrent": "y_AssetsCurrent",
    "LiabilitiesCurrent": "y_LiabilitiesCurrent",
    "CashAndCashEquivalentsAtCarryingValue": "y_CashAndCashEquivalentsAtCarryingValue",
}

SOLVENCY_REQ = ["Assets", "Liabilities", "StockholdersEquity"]
LIQUIDITY_REQ = ["AssetsCurrent", "LiabilitiesCurrent", "CashAndCashEquivalentsAtCarryingValue"]


def _safe_div(a, b):
    return a / (b + EPS)


def _to_next_value(current: float, pred: float, delta_mode: str) -> float:
    if current is None or (isinstance(current, float) and np.isnan(current)):
        return np.nan
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return np.nan

    # Ensure numeric
    try:
        cur = float(current)
        p = float(pred)
    except Exception:
        return np.nan

    if delta_mode == "logdiff":
        # pred is delta in log1p space
        cur_nonneg = max(cur, 0.0)
        return float(np.expm1(np.log1p(cur_nonneg) + p))

    if delta_mode == "pct":
        return float(cur * (1.0 + p))

    if delta_mode == "abs":
        return float(cur + p)

    raise ValueError(f"Unknown delta_mode={delta_mode}")


def _compute_ratios(row: pd.Series, prefix: str = "") -> Dict[str, float]:
    A = row.get(prefix + "Assets", np.nan)
    L = row.get(prefix + "Liabilities", np.nan)
    E = row.get(prefix + "StockholdersEquity", np.nan)
    CA = row.get(prefix + "AssetsCurrent", np.nan)
    CL = row.get(prefix + "LiabilitiesCurrent", np.nan)
    C = row.get(prefix + "CashAndCashEquivalentsAtCarryingValue", np.nan)

    return {
        "current_ratio": _safe_div(CA, CL),
        "cash_ratio": _safe_div(C, CL),
        "debt_to_assets": _safe_div(L, A),
        "equity_to_assets": _safe_div(E, A),
        "debt_to_equity": _safe_div(L, E),
        "wc_to_assets": _safe_div(CA - CL, A),
    }


def _score_solvency(r_next: Dict[str, float]) -> float:
    ea = np.clip(r_next.get("equity_to_assets", np.nan), -1, 1)
    da = np.clip(r_next.get("debt_to_assets", np.nan), 0, 5)

    s1 = 50.0 + 50.0 * np.nan_to_num(ea, nan=0.0)
    s2 = 100.0 / (1.0 + np.nan_to_num(da, nan=1.0))
    return float(np.clip(0.6 * s1 + 0.4 * s2, 0, 100))


def _score_liquidity(r_next: Dict[str, float]) -> float:
    cr = np.clip(r_next.get("current_ratio", np.nan), 0, 5)
    cashr = np.clip(r_next.get("cash_ratio", np.nan), 0, 5)
    wc = np.clip(r_next.get("wc_to_assets", np.nan), -1, 1)

    s1 = 100.0 * np.clip(cr / 2.0, 0, 1)
    s2 = 100.0 * np.clip(cashr / 0.5, 0, 1)
    s3 = 50.0 + 50.0 * wc
    return float(np.clip(0.45 * s1 + 0.35 * s2 + 0.20 * s3, 0, 100))


def _insight(title: str, detail: str, typ: str = "warning") -> Dict[str, Any]:
    return {"type": typ, "title": title, "detail": detail}


def _balance_identity_insight(A, L, E) -> Optional[Dict[str, Any]]:
    if not (np.isfinite(A) and np.isfinite(L) and np.isfinite(E)):
        return None
    gap = float(abs(A - (L + E)))
    tol = max(1e-6, 0.02 * abs(float(A)))  # 2% of Assets
    if gap > tol:
        return _insight(
            "Balance sheet does not balance",
            f"Assets != Liabilities + Equity (gap={gap:,.0f}). Check input mapping/period.",
        )
    return None


@step(enable_cache=False)
def postprocess_scores_insights_step(
    dataset_fe_path: str,
    preds_path: str,
    out_path: str,
    delta_mode: str = "logdiff",
    preds_summary: Optional[Dict[str, Any]] = None,  # used only to enforce step order
) -> Dict[str, Any]:

    df = pd.read_parquet(dataset_fe_path)
    preds = pd.read_parquet(preds_path)

    # Determine time column
    time_col = "end" if "end" in preds.columns else None
    if time_col is None:
        raise ValueError("Predictions parquet must contain an 'end' column.")

    # Robust datetime parsing before merge
    for _df in (df, preds):
        if _df[time_col].dtype == "object":
            _df[time_col] = _df[time_col].astype(str).str.strip()
        _df[time_col] = pd.to_datetime(_df[time_col], errors="coerce")

    # Merge predictions with features/base values
    merged = preds.merge(df, on=["cik", time_col], how="left", suffixes=("", "_feat"))

    # Ensure base columns exist even if missing from df
    for base_col in BASE_COLS.keys():
        if base_col not in merged.columns:
            merged[base_col] = np.nan

    # Compute next_* from predictions
    for base_col, target_name in BASE_COLS.items():
        pred_col = f"pred_{target_name}"
        if pred_col not in merged.columns:
            raise ValueError(
                f"Missing prediction column '{pred_col}'. "
                f"Available prediction columns: {[c for c in merged.columns if c.startswith('pred_')]}"
            )

        merged[f"next_{base_col}"] = [
            _to_next_value(cur, p, delta_mode)
            for cur, p in zip(merged[base_col].values, merged[pred_col].values)
        ]

    # Ratios now + next
    now_ratios = merged.apply(lambda r: _compute_ratios(r, prefix=""), axis=1, result_type="expand")
    for c in now_ratios.columns:
        merged[c] = now_ratios[c]

    next_ratios = merged.apply(lambda r: _compute_ratios(r, prefix="next_"), axis=1, result_type="expand")
    for c in next_ratios.columns:
        merged[f"next_{c}"] = next_ratios[c]

    # Determine missing fields per score type (row-wise)
    def _row_missing(cols: List[str], r: pd.Series) -> List[str]:
        missing = []
        for c in cols:
            v = r.get(c, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                missing.append(c)
        return missing

    solvency_scores: List[float] = []
    liquidity_scores: List[float] = []
    insights_all: List[List[Dict[str, Any]]] = []

    for _, r in merged.iterrows():
        ins: List[Dict[str, Any]] = []

        # optional: parser alignment flag
        if str(r.get("period_alignment_status", "")) in ("fallback_assets_end", "fallback_anchor_latest"):
            ins.append(
                _insight(
                    "Period alignment fallback used",
                    "Could not find a common quarter across all requested tags; a fallback end date was used. "
                    "Interpret liquidity/solvency with caution and check the *_end columns.",
                )
            )

        miss_solv = _row_missing(SOLVENCY_REQ, r)
        miss_liq = _row_missing(LIQUIDITY_REQ, r)

        # Solvency
        if miss_solv:
            solvency_scores.append(np.nan)
            ins.append(_insight("Solvency score unavailable", f"Missing fields: {miss_solv}."))
        else:
            bi = _balance_identity_insight(r["Assets"], r["Liabilities"], r["StockholdersEquity"])
            if bi:
                ins.append(bi)
            solvency_scores.append(_score_solvency(_compute_ratios(r, "next_")))

        # Liquidity
        if miss_liq:
            liquidity_scores.append(np.nan)
            ins.append(_insight("Liquidity score unavailable", f"Missing fields: {miss_liq}."))
        else:
            liquidity_scores.append(_score_liquidity(_compute_ratios(r, "next_")))

        insights_all.append(ins)

    merged["solvency_score"] = solvency_scores
    merged["liquidity_score"] = liquidity_scores
    merged["insights"] = insights_all

    # Save
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(outp, index=False)

    summary = {
        "rows": int(len(merged)),
        "out_path": str(outp),
        "delta_mode": delta_mode,
        "columns": list(merged.columns),
        "n_solvency_scored": int(np.isfinite(pd.Series(solvency_scores)).sum()),
        "n_liquidity_scored": int(np.isfinite(pd.Series(liquidity_scores)).sum()),
    }
    with open(str(outp.with_suffix(".json")), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
