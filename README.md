Balance Sheet ML – Asset Forecasting from SEC Filings
Project Overview

This project builds a machine learning pipeline to forecast company balance sheet Assets using historical SEC financial statements (10-K / 10-Q).
The task is formulated as a time-aware panel regression problem, where each company is modeled across time without data leakage.

The system is designed to be:

reproducible

extensible

suitable for systematic experimentation

interpretable

Objective

Target: y_Assets (next-period Assets, optionally log-scaled)

Goal:

minimize prediction error (MAE, RMSE)

preserve directional accuracy

understand which financial signals matter most

Data Sources

SEC Company Facts

Taxonomy: us-gaap

Units: USD

Forms: 10-K, 10-Q (+ amendments)

Raw filings are parsed, normalized, and transformed into a structured panel dataset:

(company × time × financial features)

Pipeline Architecture

The project is organized into modular ZenML pipelines.

1. Data Pipeline

Extract SEC filings

Write long-format parquet files per company

Memory-safe chunking

Output

data/processed/companyfacts_long/

2. Feature Pipeline

Pivot long → wide balance sheet

Enforce minimum observations per company

Align reporting periods

Output

balance_sheet_wide.parquet

3. Model Base Pipeline

Add lagged values (*_lag1)

Create prediction targets (y_*)

Add log flags (*_is_log)

Output

balance_sheet_model_base.parquet

4. Ratio Pipeline

Construct economically meaningful ratios:

liquidity

leverage

capital structure

Output

balance_sheet_model_with_ratios.parquet

5. Training Pipeline
Time-Safe Split

For each company (cik):

oldest observations → train

most recent → validation

latest → test

This prevents:

temporal leakage

cross-company contamination

Feature Engineering

Lagged balance sheet values

Financial ratios

Optional momentum & rolling statistics

Model

HistGradientBoostingRegressor

Pipeline:

Imputer → Gradient Boosted Trees

Metrics

MAE – absolute error

RMSE – penalizes large misses

Directional accuracy – sign correctness

Results Summary

Assets are strongly autoregressive

Lagged Assets dominate predictive power

Capital structure ratios add secondary signal

Directional accuracy ≈ 65–66% on test

Increasing model complexity yields diminishing returns

Interpretability

Permutation importance on validation shows:

Top drivers:

Assets_lag1

y_Assets_is_log

equity_to_assets

liab_to_assets

Most engineered features contribute marginally, validating economic intuition.

Project Status

✅ End-to-end ML pipeline
✅ Time-aware evaluation
✅ Multiple tuned configurations (A–D)
✅ Feature importance analysis

Next steps focus on:

exploratory data analysis (EDA)

feature pruning

alternative targets

robustness checks