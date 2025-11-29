#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_one(path: Path) -> pd.DataFrame:
    obj = json.load(open(path, 'r', encoding='utf-8'))
    cik = obj.get('cik') or obj.get('CIK') or obj.get('entityId') or None
    entity = obj.get('entityName') or obj.get('EntityName') or obj.get('entityNameShort') or None

    rows = []

    # Flexible facts detection
    facts = obj.get('facts') or obj.get('us-gaap') or {}
    if not facts:
        # fallback: some files put concepts at top level
        facts = {k: v for k, v in obj.items() if isinstance(v, (dict, list)) and ('-' in k or k.isidentifier())}

    if 'us-gaap' in facts and isinstance(facts['us-gaap'], dict):
        facts = facts['us-gaap']

    if isinstance(facts, dict):
        for concept, meta in facts.items():
            # Case: meta contains unit → list mapping
            if isinstance(meta, dict):
                units = meta.get('units') or meta.get('values') or {}
                if isinstance(units, dict) and units:
                    for unit_name, entries in units.items():
                        for entry in entries:
                            val = entry.get('val') or entry.get('value')
                            if val is None:
                                continue
                            rows.append({
                                'cik': cik,
                                'entity': entity,
                                'concept': concept,
                                'label': meta.get('label'),
                                'unit': unit_name,
                                'value': val,
                                'start': entry.get('start'),
                                'end': entry.get('end') or entry.get('date'),
                                'fy': entry.get('fy'),
                                'fp': entry.get('fp'),
                                'accn': entry.get('accn'),
                            })
                # case: meta is list of entries
                elif isinstance(meta, list):
                    for entry in meta:
                        val = entry.get('val') or entry.get('value')
                        if val is None:
                            continue
                        rows.append({
                            'cik': cik,
                            'entity': entity,
                            'concept': concept,
                            'label': None,
                            'unit': entry.get('unit'),
                            'value': val,
                            'start': entry.get('start'),
                            'end': entry.get('end') or entry.get('date'),
                            'fy': entry.get('fy'),
                            'fp': entry.get('fp'),
                            'accn': entry.get('accn'),
                        })
            elif isinstance(meta, list):
                for entry in meta:
                    val = entry.get('val') or entry.get('value')
                    if val is None:
                        continue
                    rows.append({
                        'cik': cik,
                        'entity': entity,
                        'concept': concept,
                        'label': None,
                        'unit': entry.get('unit'),
                        'value': val,
                        'start': entry.get('start'),
                        'end': entry.get('end') or entry.get('date'),
                        'fy': entry.get('fy'),
                        'fp': entry.get('fp'),
                        'accn': entry.get('accn'),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['end'] = pd.to_datetime(df['end'], errors='coerce')
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['end','value']).reset_index(drop=True)
    return df

def parse_folder(folder: Path, limit: int = None) -> pd.DataFrame:
    files = sorted(folder.glob('*.json'))
    if limit:
        files = files[:limit]   # only keep the first N files
    dfs = []
    for i, f in enumerate(files, 1):
        try:
            logger.info("Parsing file %d/%d: %s", i, len(files), f.name)
            df = parse_one(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.exception("Parse failed %s: %s", f, e)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to parse")
    args = parser.parse_args()

    out = parse_folder(Path(args.input), limit=args.limit)

    # Ensure CIK is string
    if "cik" in out.columns:
        out["cik"] = out["cik"].astype(str)

    out.to_parquet(args.output, index=False)
    logger.info("Saved %s", args.output)



if __name__ == "__main__":
    main()
