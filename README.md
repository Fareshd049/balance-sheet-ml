# 📊 Balance Sheet ML – Financial Risk & Directional Forecasting

## Project Overview

This project builds an **end-to-end machine learning system** to analyze and forecast **future balance-sheet dynamics** of companies using publicly available SEC filings.

Rather than predicting exact future accounting values, the system focuses on **directional changes** in key financial concepts (assets, liabilities, equity, liquidity), enabling **early warning signals** for:

- banks (credit risk & solvency monitoring)
- suppliers (counterparty risk)
- investors & regulators (financial stability)

The project is implemented using **ZenML pipelines**, ensuring modularity, reproducibility, and deployability.

---

## 🎯 Problem Statement

Given a company’s historical balance-sheet data:

> **Can we predict whether its financial position is improving or deteriorating in the next reporting period?**

Key challenges addressed:
- highly autocorrelated financial time series  
- non-stationary regimes  
- heavy-tailed changes  
- heterogeneous firm sizes  

---

## 🧠 Modeling Philosophy

- **We predict changes, not levels**
- Targets are defined as **safe log-differences** (or raw differences when needed)
- Emphasis is placed on **directional accuracy**, not precise magnitudes
- Predictions are interpreted as **risk signals**, not accounting forecasts

This approach reflects **real-world financial risk modeling practices**.

---

## 🗂️ Data Source

- **SEC Company Facts** (XBRL, US-GAAP)
- Forms used: `10-K`, `10-Q`, and amendments
- Currency filtered to USD
- Time coverage: 1980 → present
- Company-level quarterly & annual filings

---

## ⚙️ Pipeline Architecture (ZenML)

The project is organized into **independent, reproducible pipelines**:

### 1️⃣ Data Pipeline
- Extract SEC Company Facts
- Normalize to long format
- Partition by company

### 2️⃣ Feature Pipeline
- Build wide balance-sheet tables
- Align filings chronologically
- Enforce accounting consistency  
  *(Assets = Liabilities + Equity)*

### 3️⃣ Model Base Pipeline
- Create lag features (t−1)
- Filter companies with sufficient history
- Build prediction targets as safe log-differences

### 4️⃣ Ratio Feature Pipeline
- Compute structural financial ratios:
  - leverage
  - liquidity
  - capital structure
- Add derived ratio features

### 5️⃣ Training Pipeline (Multi-Target)
- Chronological split per company:
  - **Train** (historical)
  - **Validation** (near future)
  - **Test** (most recent)
- One model per target (multi-model strategy)
- Model: `HistGradientBoostingRegressor`
- Early stopping enabled

### 6️⃣ Inference Pipeline
- Predict **next-period changes**
- Output firm-level directional signals
- Ready for deployment / dashboarding

---

## 🎯 Prediction Targets

The system predicts **one-step-ahead changes** for:

### Solvency
- `y_Assets`
- `y_Liabilities`
- `y_StockholdersEquity`

### Liquidity
- `y_AssetsCurrent`
- `y_LiabilitiesCurrent`
- `y_CashAndCashEquivalentsAtCarryingValue`

All outputs represent **future changes**, not absolute values.

---

## 📈 Evaluation Metrics

For each target, evaluation is performed on **validation and test sets** using:

- **MAE** – average magnitude error
- **RMSE** – penalizes large errors
- **Directional Accuracy** – correctness of the sign (↑ / ↓)

### Key Results (Test Set – Highlights)

| Target | Directional Accuracy |
|------|----------------------|
| Assets | ~66% |
| Liabilities | ~68% |
| Equity | ~71% |
| Current Assets | ~71% |
| Current Liabilities | ~71% |
| Cash | **~81%** |

Liquidity variables, especially **cash**, show the strongest predictability.

---

## ⚠️ Interpretation & Limitations

- The model **does not predict exact future balance sheets**
- Large numerical errors can occur when reconstructing levels
- Derived ratios are interpreted as **qualitative risk indicators**
- Directional signals are **statistically robust and actionable**

This design choice is **intentional and correct** for financial risk modeling.

---

## 📊 Visualization & Deployment

The inference outputs are designed to power a **Streamlit dashboard**, featuring:

- Directional arrows with reliability indicators
- Liquidity & solvency stress patterns
- Company ranking by risk signals
- Early-warning summaries

The project is fully **deployable**, with clean separation between:
- training
- evaluation
- inference

---

## 📁 Project Structure

```text
balance-sheet-ml/
│
├── pipelines/
│   ├── data_pipeline.py
│   ├── feature_pipeline.py
│   ├── model_base_pipeline.py
│   ├── feature_ratio_pipeline.py
│   └── training_pipeline.py
│
├── steps/
│   ├── build_model_base_step.py
│   ├── add_ratio_features_step.py
│   ├── split_step.py
│   ├── train_tree_multi_step.py
│   ├── evaluate_multi_step.py
│   └── predict_latest_step.py
│
├── notebooks/
│   └── final_submission_notebook.ipynb
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── pipelines_config.yaml
├── run_pipeline.py
└── README.md




🧪 How to Run
# Install dependencies
pip install -r requirements.txt

# Run pipelines (configured via pipelines_config.yaml)
python run_pipeline.py


ZenML dashboard (optional):

zenml login --local

🧾 Academic Positioning

This project demonstrates:

correct time-series splitting

avoidance of data leakage

principled target design

interpretable financial ML

production-ready pipeline design

It is suitable for:

ML engineering coursework

applied financial ML projects

early-warning risk systems

👤 Author

Fares El Hamdaoui
End-to-End Financial Machine Learning Project

