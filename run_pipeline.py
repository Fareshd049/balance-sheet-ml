import yaml
from pipelines.data_pipeline import data_pipeline
from pipelines.feature_pipeline import feature_pipeline


def run_data_pipeline(cfg: dict) -> None:
    data_cfg = cfg["data"]

    # In many ZenML versions, calling the pipeline already executes it.
    response = data_pipeline(
        zip_path=data_cfg["zip_path"],
        extract_dir=data_cfg["extract_dir"],
        output_dir=data_cfg["output_dir"],
        max_files=int(data_cfg.get("max_files", 0)),
        chunk_rows=int(data_cfg.get("chunk_rows", 250_000)),
        include_taxonomies=data_cfg.get("include_taxonomies"),
        forms_keep=data_cfg.get("forms_keep"),
    )

    print("\n✅ Data pipeline triggered.")
    print(response)

def run_feature_pipeline(cfg: dict) -> None:
    feat_cfg = cfg["features"]

    response = feature_pipeline(
        long_dataset_dir=feat_cfg["long_dataset_dir"],
        out_path=feat_cfg["out_path"],
        unit_keep=feat_cfg.get("unit_keep", "USD"),
        forms_keep=feat_cfg.get("forms_keep"),
        min_end=feat_cfg.get("min_end", "1980-01-01"),
        max_end=feat_cfg.get("max_end"),
    )

    print("\n✅ Feature pipeline triggered.")
    print(response)

def main() -> None:
    with open("pipelines_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    active = cfg.get("pipelines", {}).get("active", "data")
    if active == "data":
        run_data_pipeline(cfg)
    elif active == "features":
        run_feature_pipeline(cfg)
    else:
        raise ValueError(f"Unknown pipeline: {active}")

if __name__ == "__main__":
    main()
