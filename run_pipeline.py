import yaml
from pipelines.data_pipeline import data_pipeline
from pipelines.feature_pipeline import feature_pipeline
from pipelines.model_base_pipeline import model_base_pipeline
from pipelines.training_pipeline import training_pipeline
from pipelines.feature_ratio_pipeline import feature_ratio_pipeline



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

def run_model_base_pipeline(cfg: dict) -> None:
    mcfg = cfg["model_base"]
    response = model_base_pipeline(
        wide_path=mcfg["wide_path"],
        out_path=mcfg["out_path"],
        concepts=mcfg.get("concepts"),
        min_obs_per_company=int(mcfg.get("min_obs_per_company", 5)),
    )
    print("\n✅ Model-base pipeline triggered.")
    print(response)

def run_training_pipeline(cfg: dict) -> None:
    tcfg = cfg["training"]

    response = training_pipeline(
        dataset_path=tcfg["dataset_path"],
        target_col=tcfg.get("target_col", "y_Assets"),
        n_test=int(tcfg.get("n_test", 2)),
        n_val=int(tcfg.get("n_val", 1)),

        # keep alpha for compatibility (unused in tree pipeline)
        alpha=float(tcfg.get("alpha", 1.0)),

        # tree params
        max_iter=int(tcfg.get("max_iter", 300)),
        learning_rate=float(tcfg.get("learning_rate", 0.05)),
        max_depth=int(tcfg.get("max_depth", 6)),
        early_stopping=bool(tcfg.get("early_stopping", True)),
        n_iter_no_change=int(tcfg.get("n_iter_no_change", 20)),
        validation_fraction=float(tcfg.get("validation_fraction", 0.1)),
    )

    print("\n✅ Training pipeline triggered.")
    print(response)



def run_ratio_pipeline(cfg: dict) -> None:
    rcfg = cfg["ratios"]
    response = feature_ratio_pipeline(
        dataset_path=rcfg["dataset_path"],
        out_path=rcfg["out_path"],
    )
    print("\n✅ Ratio pipeline triggered.")
    print(response)


def main() -> None:
    with open("pipelines_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    active = cfg.get("pipelines", {}).get("active", "data")
    if active == "data":
        run_data_pipeline(cfg)
    elif active == "features":
        run_feature_pipeline(cfg)
    elif active == "model_base":
        run_model_base_pipeline(cfg)
    elif active == "training":
        run_training_pipeline(cfg)
    elif active == "ratios":
        run_ratio_pipeline(cfg)
    else:
        raise ValueError(f"Unknown pipeline: {active}")

if __name__ == "__main__":
    main()
