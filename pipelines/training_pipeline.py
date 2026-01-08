from typing import Dict, Any, List
from zenml import pipeline

from steps.split_step import split_step
from steps.feature_engineering_step import feature_engineering_step
from steps.train_tree_multi_step import train_tree_multi_step
from steps.evaluate_multi_step import evaluate_multi_step


@pipeline(enable_cache=False)
def training_pipeline(
    dataset_path: str,
    target_cols: List[str],
    n_test: int = 2,
    n_val: int = 1,
    alpha: float = 1.0,

    # tree params
    max_iter: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    early_stopping: bool = True,
    n_iter_no_change: int = 20,
    validation_fraction: float = 0.1,
) -> Dict[str, Any]:

    # Split (no need to filter by any target here)
    train_path, val_path, test_path, split_stats = split_step(
        dataset_path=dataset_path,
        target_col=target_cols[0],  # keep compatibility with current split_step signature
        n_test=n_test,
        n_val=n_val,
    )

    # Feature engineering
    train_path_fe, val_path_fe, test_path_fe = feature_engineering_step(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
    )

    # Train one model per target
    model_paths, train_stats_json = train_tree_multi_step(
        train_path=train_path_fe,
        target_cols=target_cols,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
        validation_fraction=validation_fraction,
    )

    # Evaluate all targets
    metrics_by_target = evaluate_multi_step(
        model_paths=model_paths,
        val_path=val_path_fe,
        test_path=test_path_fe,
        compute_importance=False,  # set True only if you want and accept runtime
    )

    return {
        "split": split_stats,
        "train_stats_json": train_stats_json,
        "model_paths": model_paths,
        "metrics_by_target": metrics_by_target,
    }
