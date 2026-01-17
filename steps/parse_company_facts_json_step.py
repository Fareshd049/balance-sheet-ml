from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import pandas as pd
import numpy as np
from zenml import step

REQUIRED_TAGS = [
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "CashAndCashEquivalentsAtCarryingValue",
]

CORE_TAGS = ["Assets", "Liabilities", "StockholdersEquity"]
LIQ_TAGS  = ["AssetsCurrent", "LiabilitiesCurrent", "CashAndCashEquivalentsAtCarryingValue"]

def _usd_list(tag_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    return tag_block.get("units", {}).get("USD", []) or []

def _ends_set(tag_block: Dict[str, Any]) -> set:
    return set([x.get("end") for x in _usd_list(tag_block) if x.get("end")])

def _latest_end(tag_block: Dict[str, Any]) -> Optional[str]:
    usd = _usd_list(tag_block)
    if not usd:
        return None
    latest = max(usd, key=lambda x: x.get("end", "0000-00-00"))
    return latest.get("end")

def _value_for_end(tag_block: Dict[str, Any], chosen_end: str) -> Tuple[Optional[float], Optional[str]]:
    for item in _usd_list(tag_block):
        if item.get("end") == chosen_end:
            return item.get("val"), item.get("end")
    return None, None

def _latest_value(tag_block: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    usd = _usd_list(tag_block)
    if not usd:
        return None, None
    latest = max(usd, key=lambda x: x.get("end", "0000-00-00"))
    return latest.get("val"), latest.get("end")

def _choose_best_common_end(facts_data: Dict[str, Any], anchor_tag: str = "Assets") -> Tuple[str, str]:
    """
    Returns (chosen_end, strategy).
    strategy: "intersection_all", "intersection_core", "fallback_anchor_latest"
    """
    # If anchor missing, pick first available core tag
    if anchor_tag not in facts_data:
        anchor_tag = next((t for t in CORE_TAGS if t in facts_data), None)
        if anchor_tag is None:
            raise ValueError("None of CORE_TAGS exist in this JSON.")

    # Build ends per tag
    ends_by_tag = {t: _ends_set(facts_data[t]) for t in REQUIRED_TAGS if t in facts_data}

    # Candidate ends: go from latest anchor ends backward
    anchor_ends = sorted(list(ends_by_tag.get(anchor_tag, set())), reverse=True)
    if not anchor_ends:
        raise ValueError(f"Anchor tag '{anchor_tag}' has no USD facts.")

    # 1) Try strict intersection (all required tags present at same end)
    for e in anchor_ends:
        ok_all = all((t in ends_by_tag and e in ends_by_tag[t]) for t in REQUIRED_TAGS if t in facts_data)
        if ok_all:
            return e, "intersection_all"

    # 2) Try core intersection (Assets/Liabilities/Equity present at same end)
    for e in anchor_ends:
        ok_core = all((t in ends_by_tag and e in ends_by_tag[t]) for t in CORE_TAGS if t in facts_data)
        if ok_core:
            return e, "intersection_core"

    # 3) Fallback to anchor latest
    return anchor_ends[0], "fallback_anchor_latest"

@step(enable_cache=False)
def parse_company_facts_json_step(
    json_path: str,
    out_parquet_path: str,
    anchor_tag: str = "Assets",
) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "facts" in data:
        cik = str(data.get("cik", "")).strip()
        entity_name = data.get("entityName", "")
        facts_data = data.get("facts", {}).get("us-gaap", {}) or {}
        if not facts_data:
            raise ValueError("No 'us-gaap' facts found in JSON")
    else:
        facts_data = data
        cik = Path(json_path).stem.replace("CIK", "")
        entity_name = ""

    chosen_end, strategy = _choose_best_common_end(facts_data, anchor_tag=anchor_tag)

    row: Dict[str, Any] = {}
    row["cik"] = cik if cik else Path(json_path).stem.replace("CIK", "")
    row["end"] = str(chosen_end).strip()
    row["entity_name"] = entity_name
    row["period_alignment_status"] = strategy

    # extract values for chosen_end; if missing -> fallback to latest and flag it
    for tag in REQUIRED_TAGS:
        if tag not in facts_data:
            row[tag] = np.nan
            row[f"{tag}_end"] = None
            row[f"{tag}_missing_for_end"] = row["end"]
            continue

        val, end = _value_for_end(facts_data[tag], row["end"])
        if val is None:
            # fallback
            val2, end2 = _latest_value(facts_data[tag])
            row[tag] = float(val2) if val2 is not None else np.nan
            row[f"{tag}_end"] = end2
            row[f"{tag}_missing_for_end"] = row["end"]
        else:
            row[tag] = float(val) if val is not None else np.nan
            row[f"{tag}_end"] = end
            row[f"{tag}_missing_for_end"] = None

    # ensure end is parseable
    row["end"] = pd.to_datetime(row["end"], errors="coerce")
    if pd.isna(row["end"]):
        raise ValueError(f"Could not parse chosen end date: {chosen_end}")

    df = pd.DataFrame([row])
    Path(out_parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet_path, index=False)
    return out_parquet_path
