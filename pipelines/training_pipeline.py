from typing import Dict, Any
from zenml import pipeline

from steps.split_step import split_step
from steps.feature_engineering_step import feature_engineering_step
from steps.train_tree_step import train_tree_step
from steps.evaluation_step import evaluate_step


@pipeline(enable_cache=False)
def training_pipeline(
    dataset_path: str,
    target_col: str = "y_Assets",
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

    train_path, val_path, test_path, split_stats = split_step(
        dataset_path=dataset_path,
        target_col=target_col,
        n_test=n_test,
        n_val=n_val,
    )

    train_path_fe, val_path_fe, test_path_fe = feature_engineering_step(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
    )

    model_path, train_stats_json = train_tree_step(
        train_path=train_path_fe,
        target_col=target_col,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
        validation_fraction=validation_fraction,
    )

    metrics = evaluate_step(
        model_path=model_path,
        val_path=val_path_fe,
        test_path=test_path_fe,
    )

    return {
        "split": split_stats,
        "train_stats_json": train_stats_json,
        "model_path": model_path,
        "metrics": metrics,
    }
