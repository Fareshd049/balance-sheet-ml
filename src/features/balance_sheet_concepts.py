# src/features/balance_sheet_concepts.py

BALANCE_SHEET_CONCEPTS = [
    # Core totals
    "Assets",
    "AssetsCurrent",
    "AssetsNoncurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "LiabilitiesNoncurrent",
    "StockholdersEquity",
    "LiabilitiesAndStockholdersEquity",

    # Cash & working capital
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "AccountsReceivableNetCurrent",
    "InventoryNet",
    "PrepaidExpenseCurrent",

    # PP&E and intangibles
    "PropertyPlantAndEquipmentNet",
    "Goodwill",
    "IntangibleAssetsNetExcludingGoodwill",

    # Debt
    "DebtCurrent",
    "LongTermDebt",
    "LongTermDebtCurrent",

    # Payables / accruals
    "AccountsPayableCurrent",
    "AccruedLiabilitiesCurrent",

    # Equity components (optional but useful)
    "RetainedEarningsAccumulatedDeficit",
    "CommonStockValue",
    "AdditionalPaidInCapital",
    "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
]
