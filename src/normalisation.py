import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 1. Drop des colonnes trop vides
def drop_sparse_features(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    frac_present = df.notna().mean()
    keep_cols = frac_present[frac_present > threshold].index.tolist()
    meta_cols = ["cik","entity","fy","fp","end","accn"]
    return df[meta_cols + [c for c in keep_cols if c not in meta_cols]]

def time_based_imputation(df: pd.DataFrame) -> pd.DataFrame:
    def fill_group(g):
        g = g.sort_values("end")
        g = g.ffill().bfill()
        # Sélectionner uniquement les colonnes numériques
        num_cols = g.select_dtypes(include=[np.number, "Int64"]).columns
        if len(num_cols) > 0:
            # Convertir en float pour supporter NaN + interpolation
            g[num_cols] = g[num_cols].astype(float)
            g[num_cols] = g[num_cols].interpolate(method="linear", limit_direction="both")
        return g
    return df.groupby("cik").apply(fill_group).reset_index(drop=True)


# 3. Apply enriched accounting rules
def apply_accounting_rules(df: pd.DataFrame) -> pd.DataFrame:
    # Assets = Liabilities + Equity
    if {"Assets","Liabilities","StockholdersEquity"}.issubset(df.columns):
        mask = df["Assets"].isna()
        df.loc[mask, "Assets"] = df.loc[mask, "Liabilities"] + df.loc[mask, "StockholdersEquity"]

    # Equity = Assets - Liabilities
    if {"Assets","Liabilities","StockholdersEquity"}.issubset(df.columns):
        mask = df["StockholdersEquity"].isna()
        df.loc[mask, "StockholdersEquity"] = df.loc[mask, "Assets"] - df.loc[mask, "Liabilities"]

    # Cash Flow = Operating + Investing + Financing
    if {"NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInFinancingActivities",
        "CashAndCashEquivalentsAtCarryingValue"}.issubset(df.columns):
        mask = df["CashAndCashEquivalentsAtCarryingValue"].isna()
        df.loc[mask, "CashAndCashEquivalentsAtCarryingValue"] = (
            df.loc[mask, "NetCashProvidedByUsedInOperatingActivities"] +
            df.loc[mask, "NetCashProvidedByUsedInInvestingActivities"] +
            df.loc[mask, "NetCashProvidedByUsedInFinancingActivities"]
        )



    return df

# 4. Optimized MICE
def mice_imputation(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    exclude = set(exclude_cols or ["cik","entity","fy","fp","end","accn"])
    incomplete_cols = [c for c in df.columns 
                       if c not in exclude and df[c].isna().any() and pd.api.types.is_numeric_dtype(df[c])]
    
    if not incomplete_cols:
        logging.info("Aucune colonne incomplète à imputer avec MICE.")
        return df
    
    logging.info(f"Colonnes imputées avec MICE : {incomplete_cols}")
    imputer = IterativeImputer(
        estimator=LinearRegression(),
        max_iter=3,
        skip_complete=True,
        random_state=42
    )
    df[incomplete_cols] = imputer.fit_transform(df[incomplete_cols])
    return df

def log_transform(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    exclude = set(exclude_cols or ["cik","entity","fy","fp","end","accn"])
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # Convertir en float pour éviter les problèmes avec Int64
    df[num_cols] = df[num_cols].astype(float)
    
    # Appliquer log1p vectorisé
    df[num_cols] = np.sign(df[num_cols]) * np.log1p(np.abs(df[num_cols]))
    
    return df


# 🔹 Pipeline complet
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Step 1 — Drop des colonnes trop vides (seuil 1%)…")
    df = drop_sparse_features(df, threshold=0.1)

    logging.info("Step 2 — Time-based imputation…")
    df = time_based_imputation(df)

    logging.info("Step 3 — Apply accounting rules…")
    df = apply_accounting_rules(df)

    logging.info("Step 4 — Optimized MICE…")
    df = mice_imputation(df)

    logging.info("Step 5 — Log transform…")
    df = log_transform(df)

    logging.info("Pipeline terminé ✅")
    return df

# 🔹 Exemple d'utilisation
if __name__ == "__main__":
    parquet_path = "data/processed/wide_0.parquet"

    # Charger directement toutes les colonnes
    wide = pd.read_parquet(parquet_path)
    logging.info(f"Shape avant normalisation : {wide.shape}")

    # Pipeline complet
    wide_norm = normalize_dataframe(wide)
    logging.info(f"Shape après normalisation : {wide_norm.shape}")

    # Sauvegarde
    wide_norm.to_parquet("data/processed/wide_norm_0.parquet", index=False)
    logging.info("wide_norm sauvegardé dans data/processed/wide_norm_0.parquet ✅")
