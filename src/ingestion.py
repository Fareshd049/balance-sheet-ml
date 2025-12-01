import json
import pandas as pd
from pathlib import Path

def load_json_files(files) -> pd.DataFrame:
    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as infile:
            obj = json.load(infile)

        cik = str(obj.get("cik")) if obj.get("cik") is not None else None
        entity = obj.get("entityName")

        facts = obj.get("facts", {}).get("us-gaap", {})
        for concept, meta in facts.items():
            units = meta.get("units", {})
            for unit, entries in units.items():
                for e in entries:
                    rows.append({
                        "cik": cik,
                        "entity": entity,
                        "concept": concept,
                        "unit": unit,
                        "value": e.get("val"),
                        "fy": e.get("fy"),
                        "fp": e.get("fp"),
                        "form": e.get("form"),
                        "accn": e.get("accn"),
                        "end": e.get("end"),
                    })
    df = pd.DataFrame(rows)

    if not df.empty:
        df["cik"] = df["cik"].astype("string")
        df["entity"] = df["entity"].astype("string")
        df["concept"] = df["concept"].astype("string")
        df["unit"] = df["unit"].astype("string")
        df["fp"] = df["fp"].astype("string")
        df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["form"] = df["form"].astype("string")
        df["accn"] = df["accn"].astype("string")
        df["end"] = pd.to_datetime(df["end"], errors="coerce")

    return df


if __name__ == "__main__":
    folder = Path("data\\raw")
    interim = Path("data\\interim")
    interim.mkdir(parents=True, exist_ok=True)

    files = list(folder.glob("*.json"))
    total_files = len(files)
    batch_size = 1000

    print(f"📂 Found {total_files} JSON files to process")

    for i in range(0, total_files, batch_size):
        out_file = interim / f"clean_part{i//batch_size}.parquet"

        if out_file.exists():
            print(f"⏩ Skipping batch {i//batch_size}, {out_file} already exists")
            continue

        batch_files = files[i:i+batch_size]
        df = load_json_files(batch_files)

        if df.empty:
            print(f"⚠️ Batch {i//batch_size} produced no rows, skipping")
            continue

        df.to_parquet(out_file, index=False)

        processed = min(i+batch_size, total_files)
        print(f"✅ Batch {i//batch_size}: saved {len(df)} rows → {out_file} "
              f"({processed}/{total_files} files processed)")
