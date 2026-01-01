

from pathlib import Path
import zipfile
from typing import Dict

from zenml import step


@step
def extract_companyfacts_step(
    zip_path: str,
    extract_dir: str,
) -> Dict[str, str]:
    """
    Unzip companyfacts.zip into a directory of JSON files.

    Idempotent:
    - If extract_dir already contains JSON files, extraction is skipped.

    Parameters
    ----------
    zip_path : str
        Path to companyfacts.zip (e.g. data/raw/companyfacts.zip)
    extract_dir : str
        Target directory for extracted JSON files (e.g. data/raw/companyfacts)

    Returns
    -------
    Dict[str, str]
        Paths used (for logging / downstream steps)
    """
    zip_path_p = Path(zip_path)
    extract_dir_p = Path(extract_dir)

    if not zip_path_p.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path_p}")

    extract_dir_p.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_jsons = list(extract_dir_p.glob("*.json"))
    if existing_jsons:
        return {
            "status": "skipped",
            "zip_path": str(zip_path_p),
            "extract_dir": str(extract_dir_p),
            "json_files_found": str(len(existing_jsons)),
        }

    # Extract
    with zipfile.ZipFile(zip_path_p, "r") as zf:
        zf.extractall(extract_dir_p)

    json_count = len(list(extract_dir_p.glob("*.json")))

    return {
        "status": "extracted",
        "zip_path": str(zip_path_p),
        "extract_dir": str(extract_dir_p),
        "json_files_extracted": str(json_count),
    }
