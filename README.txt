ML-Balance-Sheet-Prediction
Automate and improve the accuracy of a company’s balance sheet structure forecasts by developing a machine learning model capable of simultaneously leveraging: • The financial history specific to the target company (balance sheet, income statement, cash flows, etc.), • The common dynamics observed within a large sample of companies.

Project Structure : 
balnce-sheet-ml/
│
├── data/
│   ├── raw/               # raw JSON filings (local only, .gitignore)
│   ├── interim/           # intermediate cleaned files
│   └── processed/         # final parquet/csv ready for ML
│
├── notebooks/             # Jupyter notebooks for EDA, prototyping
│   ├── 01_ingestion.ipynb
│   ├── 02_cleaning.ipynb
│   └── 03_normalization.ipynb
│
├── src/                   # reusable Python modules
│   ├── ingestion.py       # load JSON → DataFrame
│   ├── cleaning.py        # drop duplicates, fix types
│   └── normalization.py   # scale, standardize, reconcile
│
├── tests/                 # unit tests for functions
│
├── scripts/               # CLI scripts
│   └── clean_xbrl.py      # your parsing script
│
├── requirements.txt       # dependencies
├── README.md              # project overview
└── .gitignore             # ignore venv, raw data, cache



