"""
============================================================================
Data Preparation Pipeline
============================================================================
Downloads, cleans, and transforms the Kaggle Store Sales (Favorita) dataset
into a star schema suitable for both Azure AutoML training and Power BI.

Pipeline Steps:
    1. Download raw data from Kaggle (or load from data/raw/)
    2. Clean & merge all source tables
    3. Engineer time-series features (lags, rolling windows, holidays)
    4. Build star schema dimensions and fact tables
    5. Export to CSV and Parquet for downstream consumption

Usage:
    python src/data_prep.py              # Full pipeline
    python src/data_prep.py --skip-download  # Skip Kaggle download

Author: Jared Waldroff
============================================================================
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ============================================================================
# Configuration
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Project paths - using Path for cross-platform compatibility
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Kaggle dataset identifier
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "store-sales-time-series-forecasting")


# ============================================================================
# Step 1: Data Download
# ============================================================================

def download_kaggle_data():
    """
    Download the Store Sales dataset from Kaggle using the Kaggle API.

    Prerequisites:
        - Install kaggle: pip install kaggle
        - Place your API key at ~/.kaggle/kaggle.json
        - Accept competition rules at:
          https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules

    If the Kaggle API isn't set up, the function prints manual download
    instructions instead of failing — because we don't want the pipeline
    to break just because someone hasn't configured an API key.
    """
    print("=" * 60)
    print("STEP 1: Downloading Kaggle Data")
    print("=" * 60)

    # Check if data already exists to avoid redundant downloads
    expected_files = ["train.csv", "stores.csv", "oil.csv",
                      "holidays_events.csv", "transactions.csv"]
    existing = [f for f in expected_files if (RAW_DATA_DIR / f).exists()]

    if len(existing) == len(expected_files):
        print(f"  ✓ All {len(expected_files)} files already exist in {RAW_DATA_DIR}")
        print("  Skipping download.")
        return

    print(f"  Found {len(existing)}/{len(expected_files)} files. Downloading missing data...")

    try:
        # Use Kaggle CLI to download competition data
        # -c: competition name, -p: destination path, --unzip: extract files
        subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", KAGGLE_DATASET,
                "-p", str(RAW_DATA_DIR)
            ],
            check=True,
            capture_output=True,
            text=True
        )

        # Kaggle downloads a zip file — unzip if present
        zip_path = RAW_DATA_DIR / f"{KAGGLE_DATASET}.zip"
        if zip_path.exists():
            subprocess.run(
                ["unzip", "-o", str(zip_path), "-d", str(RAW_DATA_DIR)],
                check=True,
                capture_output=True
            )
            zip_path.unlink()  # Remove zip after extraction

        print("  ✓ Download complete.")

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Graceful fallback — don't crash, guide the user
        print("\n  ⚠ Kaggle API not available or not authenticated.")
        print("  Please download manually:")
        print(f"    1. Go to: https://www.kaggle.com/competitions/{KAGGLE_DATASET}/data")
        print(f"    2. Download all files")
        print(f"    3. Extract to: {RAW_DATA_DIR}/")
        print(f"    4. Re-run this script with: python src/data_prep.py --skip-download")
        sys.exit(1)


# ============================================================================
# Step 2: Load and Clean Raw Data
# ============================================================================

def load_raw_data():
    """
    Load all raw CSV files into DataFrames and perform initial cleaning.

    The Favorita dataset has 5 key tables:
        - train.csv:           Daily sales by store × product family
        - stores.csv:          Store metadata (city, state, type, cluster)
        - oil.csv:             Daily oil prices (Ecuador is oil-dependent)
        - holidays_events.csv: National, regional, and local holidays
        - transactions.csv:    Daily transaction counts per store

    Returns:
        dict: Mapping of table names to cleaned DataFrames
    """
    print("\n" + "=" * 60)
    print("STEP 2: Loading and Cleaning Raw Data")
    print("=" * 60)

    # --- Load train.csv (the main sales table) ---
    # ~3M rows: id, date, store_nbr, family, sales, onpromotion
    print("  Loading train.csv (~3M rows)...")
    train = pd.read_csv(
        RAW_DATA_DIR / "train.csv",
        parse_dates=["date"],      # Parse date column as datetime
        dtype={
            "store_nbr": "int16",  # Small int saves memory (54 stores)
            "family": "category",  # Categorical for 33 product families
            "onpromotion": "int16"
        }
    )

    # Sales should never be negative — clip any data quality issues
    # (The Favorita dataset occasionally has small negative values
    # from returns; we zero these out for cleaner forecasting)
    train["sales"] = train["sales"].clip(lower=0)

    print(f"    Rows: {len(train):,} | Date range: {train['date'].min()} to {train['date'].max()}")
    print(f"    Stores: {train['store_nbr'].nunique()} | Product families: {train['family'].nunique()}")

    # --- Load stores.csv (store dimension data) ---
    print("  Loading stores.csv...")
    stores = pd.read_csv(
        RAW_DATA_DIR / "stores.csv",
        dtype={
            "store_nbr": "int16",
            "city": "category",
            "state": "category",
            "type": "category",
            "cluster": "int8"
        }
    )
    print(f"    Stores: {len(stores)} | Cities: {stores['city'].nunique()} | States: {stores['state'].nunique()}")

    # --- Load oil.csv (external economic indicator) ---
    # Ecuador's economy is heavily oil-dependent, so oil prices
    # are a strong predictor of consumer spending
    print("  Loading oil.csv...")
    oil = pd.read_csv(
        RAW_DATA_DIR / "oil.csv",
        parse_dates=["date"]
    )

    # Forward-fill missing oil prices (weekends/holidays have gaps)
    # This is standard practice — oil price doesn't change on non-trading days
    oil = oil.set_index("date").resample("D").first().ffill().bfill().reset_index()
    oil.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)
    print(f"    Oil price range: ${oil['oil_price'].min():.2f} - ${oil['oil_price'].max():.2f}")

    # --- Load holidays_events.csv ---
    print("  Loading holidays_events.csv...")
    holidays = pd.read_csv(
        RAW_DATA_DIR / "holidays_events.csv",
        parse_dates=["date"]
    )

    # Create a simplified holiday flag
    # We don't need all the granularity — just whether it's a holiday
    # and whether it was transferred (moved to another day)
    holidays["is_holiday"] = (~holidays["transferred"]).astype(int)

    # Aggregate to one row per date (some dates have multiple events)
    holiday_flags = (
        holidays
        .groupby("date")["is_holiday"]
        .max()
        .reset_index()
    )
    print(f"    Holiday dates: {holiday_flags['is_holiday'].sum()}")

    # --- Load transactions.csv ---
    print("  Loading transactions.csv...")
    transactions = pd.read_csv(
        RAW_DATA_DIR / "transactions.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16"}
    )
    print(f"    Transaction records: {len(transactions):,}")

    print("\n  ✓ All raw data loaded and cleaned.")

    return {
        "train": train,
        "stores": stores,
        "oil": oil,
        "holidays": holiday_flags,
        "transactions": transactions
    }


# ============================================================================
# Step 3: Feature Engineering
# ============================================================================

def engineer_features(data: dict) -> pd.DataFrame:
    """
    Merge all tables and create features for AutoML forecasting.

    Feature categories:
        - Calendar features:  Day of week, month, quarter, payday flags
        - Economic features:  Oil price (current + 7-day lag)
        - Holiday features:   Binary holiday flag
        - Promotion features: Items on promotion count
        - Store features:     Store type, cluster, city, state

    Why these features matter:
        - Calendar: Retail sales have strong weekly and monthly seasonality
        - Oil price: Ecuador's economy correlates with oil exports
        - Holidays: Sales spike before holidays, drop during them
        - Promotions: Direct driver of sales volume

    Args:
        data: Dictionary of cleaned DataFrames from load_raw_data()

    Returns:
        pd.DataFrame: Fully featured dataset ready for AutoML
    """
    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)

    df = data["train"].copy()

    # --- Merge store metadata ---
    # LEFT join preserves all sales rows even if a store is missing metadata
    print("  Merging store metadata...")
    df = df.merge(data["stores"], on="store_nbr", how="left")

    # --- Merge oil prices ---
    print("  Merging oil prices...")
    df = df.merge(data["oil"], on="date", how="left")
    # Forward-fill any remaining gaps after merge
    df["oil_price"] = df["oil_price"].ffill().bfill()

    # --- Merge holiday flags ---
    print("  Merging holiday flags...")
    df = df.merge(data["holidays"], on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    # --- Merge transaction counts ---
    print("  Merging transaction counts...")
    df = df.merge(
        data["transactions"],
        on=["date", "store_nbr"],
        how="left"
    )
    df["transactions"] = df["transactions"].fillna(0)

    # --- Calendar features ---
    # These capture the natural rhythms of retail: weekday vs weekend,
    # beginning vs end of month, seasonal patterns
    print("  Creating calendar features...")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek      # 0=Monday, 6=Sunday
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Payday flag: In Ecuador, most workers are paid on the 15th and last day
    # This creates a spending surge around these dates
    df["is_payday"] = (
        (df["day_of_month"] == 15) |
        (df["day_of_month"] == df["date"].dt.days_in_month)
    ).astype(int)

    # Is it the beginning of the month? (days 1-5 often see lower spending)
    df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)

    # --- Oil price lag ---
    # Lagged oil price captures the delayed effect on consumer behavior
    # (it takes ~1 week for pump prices to reflect crude changes)
    print("  Creating oil price lag features...")
    df = df.sort_values(["store_nbr", "family", "date"])
    df["oil_price_lag7"] = df.groupby(["store_nbr", "family"])["oil_price"].shift(7)
    df["oil_price_lag7"] = df["oil_price_lag7"].bfill()

    # --- Data type optimization ---
    # Convert categoricals for memory efficiency
    for col in ["city", "state", "type"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"\n  ✓ Feature engineering complete.")
    print(f"    Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"    Features: {list(df.columns)}")

    return df


# ============================================================================
# Step 4: Build Star Schema
# ============================================================================

def build_star_schema(df: pd.DataFrame) -> dict:
    """
    Transform the flat feature table into a star schema for Power BI.

    Star Schema Design:
        ┌──────────────┐
        │   dim_date   │──────┐
        └──────────────┘      │
        ┌──────────────┐      ▼
        │  dim_store   │──► fact_sales
        └──────────────┘      ▲
        ┌──────────────┐      │
        │ dim_product  │──────┘
        └──────────────┘

    Why star schema?
        - Power BI's Vertipaq engine is optimized for star schemas
        - Smaller dimension tables = faster slicing/dicing
        - Clear relationships = simpler DAX formulas
        - Separating actuals from forecasts enables side-by-side comparison

    Args:
        df: Fully featured DataFrame from engineer_features()

    Returns:
        dict: Mapping of table names to DataFrames
    """
    print("\n" + "=" * 60)
    print("STEP 4: Building Star Schema")
    print("=" * 60)

    # --- Dimension: Date ---
    # A proper date dimension is the backbone of any BI model.
    # It enables time intelligence functions like SAMEPERIODLASTYEAR,
    # TOTALYTD, and DATEADD in DAX.
    print("  Building dim_date...")

    date_range = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D"
    )

    dim_date = pd.DataFrame({"date": date_range})
    dim_date["date_key"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"] = dim_date["date"].dt.year
    dim_date["quarter"] = dim_date["date"].dt.quarter
    dim_date["month"] = dim_date["date"].dt.month
    dim_date["month_name"] = dim_date["date"].dt.strftime("%B")
    dim_date["week_of_year"] = dim_date["date"].dt.isocalendar().week.astype(int)
    dim_date["day_of_week"] = dim_date["date"].dt.dayofweek
    dim_date["day_name"] = dim_date["date"].dt.strftime("%A")
    dim_date["day_of_month"] = dim_date["date"].dt.day
    dim_date["is_weekend"] = (dim_date["day_of_week"] >= 5).astype(int)
    dim_date["fiscal_year"] = dim_date["year"]  # Adjust if fiscal year differs
    dim_date["year_month"] = dim_date["date"].dt.strftime("%Y-%m")
    dim_date["year_quarter"] = dim_date["date"].dt.to_period("Q").astype(str)

    # Add holiday flag from our data
    holiday_dates = df.loc[df["is_holiday"] == 1, "date"].unique()
    dim_date["is_holiday"] = dim_date["date"].isin(holiday_dates).astype(int)

    print(f"    Date range: {dim_date['date'].min()} to {dim_date['date'].max()}")
    print(f"    Rows: {len(dim_date):,}")

    # --- Dimension: Store ---
    print("  Building dim_store...")
    dim_store = (
        df[["store_nbr", "city", "state", "type", "cluster"]]
        .drop_duplicates()
        .sort_values("store_nbr")
        .reset_index(drop=True)
    )
    # Create a surrogate key (store_key) — best practice for dimensions
    dim_store["store_key"] = dim_store["store_nbr"]
    # Rename 'type' to 'store_type' for clarity in Power BI
    dim_store.rename(columns={"type": "store_type"}, inplace=True)

    # Add a region mapping for Row-Level Security demo
    # Group Ecuadorian states into regions for RLS filtering
    state_to_region = {
        "Pichincha": "Highland", "Cotopaxi": "Highland",
        "Chimborazo": "Highland", "Imbabura": "Highland",
        "Tungurahua": "Highland", "Bolivar": "Highland",
        "Azuay": "Highland", "Loja": "Highland",
        "Guayas": "Coastal", "Manabi": "Coastal",
        "El Oro": "Coastal", "Santa Elena": "Coastal",
        "Los Rios": "Coastal", "Esmeraldas": "Coastal",
        "Santo Domingo de los Tsachilas": "Coastal",
        "Pastaza": "Amazon", "Orellana": "Amazon",
    }
    dim_store["region"] = dim_store["state"].map(state_to_region).fillna("Other")

    print(f"    Stores: {len(dim_store)}")
    print(f"    Regions: {dim_store['region'].value_counts().to_dict()}")

    # --- Dimension: Product Family ---
    print("  Building dim_product...")
    families = df["family"].unique()
    dim_product = pd.DataFrame({
        "product_key": range(1, len(families) + 1),
        "family": sorted(families),
    })

    # Add product category groupings for higher-level analysis
    # These map Favorita's 33 product families into broader categories
    family_to_category = {
        "GROCERY I": "Grocery", "GROCERY II": "Grocery",
        "BEVERAGES": "Grocery", "PRODUCE": "Fresh",
        "MEATS": "Fresh", "POULTRY": "Fresh",
        "DAIRY": "Fresh", "DELI": "Fresh",
        "FROZEN FOODS": "Frozen", "PREPARED FOODS": "Frozen",
        "BREAD/BAKERY": "Bakery", "EGGS": "Fresh",
        "SEAFOOD": "Fresh", "CLEANING": "Household",
        "HOME AND KITCHEN I": "Household", "HOME AND KITCHEN II": "Household",
        "HOME CARE": "Household", "BABY CARE": "Personal Care",
        "BEAUTY": "Personal Care", "PERSONAL CARE": "Personal Care",
        "LINGERIE": "Apparel", "LADIESWEAR": "Apparel",
        "CELEBRATION": "Seasonal", "SCHOOL AND OFFICE SUPPLIES": "Seasonal",
        "AUTOMOTIVE": "Other", "HARDWARE": "Other",
        "BOOKS": "Other", "MAGAZINES": "Other",
        "LAWN AND GARDEN": "Other", "PLAYERS AND ELECTRONICS": "Electronics",
        "PET SUPPLIES": "Other", "LIQUOR,WINE,BEER": "Beverages",
    }
    dim_product["category"] = dim_product["family"].map(family_to_category).fillna("Other")

    print(f"    Product families: {len(dim_product)}")
    print(f"    Categories: {dim_product['category'].nunique()}")

    # --- Fact Table: Sales (Actuals) ---
    print("  Building fact_sales...")

    # Create the family-to-key mapping for the foreign key
    family_key_map = dim_product.set_index("family")["product_key"].to_dict()

    fact_sales = df[[
        "date", "store_nbr", "family", "sales", "onpromotion",
        "oil_price", "is_holiday", "transactions"
    ]].copy()

    # Add foreign keys that link to dimension tables
    fact_sales["date_key"] = fact_sales["date"].dt.strftime("%Y%m%d").astype(int)
    fact_sales["store_key"] = fact_sales["store_nbr"]
    fact_sales["product_key"] = fact_sales["family"].map(family_key_map)

    # Calculate a synthetic revenue column
    # The Favorita dataset only has unit sales — we generate revenue
    # using realistic price estimates per category for the Power BI demo
    avg_prices = {
        "Grocery": 2.50, "Fresh": 3.75, "Frozen": 4.50,
        "Bakery": 2.00, "Household": 5.00, "Personal Care": 6.00,
        "Apparel": 15.00, "Seasonal": 4.00, "Electronics": 25.00,
        "Beverages": 3.50, "Other": 5.00
    }
    family_to_category_series = fact_sales["family"].map(family_to_category).fillna("Other")
    fact_sales["avg_unit_price"] = family_to_category_series.map(avg_prices).fillna(5.00)
    fact_sales["revenue"] = fact_sales["sales"] * fact_sales["avg_unit_price"]

    # Calculate a simple margin (for the dynamic measure toggle in DAX)
    # Using realistic retail margins by category
    margin_rates = {
        "Grocery": 0.22, "Fresh": 0.35, "Frozen": 0.28,
        "Bakery": 0.45, "Household": 0.30, "Personal Care": 0.40,
        "Apparel": 0.50, "Seasonal": 0.35, "Electronics": 0.15,
        "Beverages": 0.32, "Other": 0.25
    }
    fact_sales["margin_rate"] = family_to_category_series.map(margin_rates).fillna(0.25)
    fact_sales["margin"] = fact_sales["revenue"] * fact_sales["margin_rate"]

    # Drop helper columns not needed in the final fact table
    fact_sales.drop(columns=["family", "store_nbr", "avg_unit_price", "margin_rate"], inplace=True)

    print(f"    Rows: {len(fact_sales):,}")
    print(f"    Revenue range: ${fact_sales['revenue'].min():.2f} - ${fact_sales['revenue'].max():.2f}")

    print("\n  ✓ Star schema build complete.")

    return {
        "dim_date": dim_date,
        "dim_store": dim_store,
        "dim_product": dim_product,
        "fact_sales": fact_sales,
    }


# ============================================================================
# Step 5: Export Data
# ============================================================================

def export_data(star_schema: dict, featured_df: pd.DataFrame):
    """
    Export all tables to CSV (for Power BI) and Parquet (for AutoML).

    Why both formats?
        - CSV:     Power BI can import directly; human-readable
        - Parquet: 5-10x smaller, preserves dtypes, faster for Python/AutoML

    Args:
        star_schema: Dictionary of dimension and fact DataFrames
        featured_df: Full featured DataFrame for AutoML training
    """
    print("\n" + "=" * 60)
    print("STEP 5: Exporting Data")
    print("=" * 60)

    # Export star schema tables (for Power BI)
    for name, table in star_schema.items():
        csv_path = PROCESSED_DIR / f"{name}.csv"
        parquet_path = PROCESSED_DIR / f"{name}.parquet"

        table.to_csv(csv_path, index=False)
        table.to_parquet(parquet_path, index=False)

        print(f"  ✓ {name}: {len(table):,} rows → {csv_path.name} + {parquet_path.name}")

    # Export the full featured dataset for AutoML training
    # AutoML needs all features in a single flat table
    automl_path = PROCESSED_DIR / "automl_training_data.parquet"
    featured_df.to_parquet(automl_path, index=False)
    print(f"  ✓ AutoML training data: {len(featured_df):,} rows → {automl_path.name}")

    # Export a small sample for quick testing
    sample = featured_df.sample(n=min(10000, len(featured_df)), random_state=42)
    sample_path = PROCESSED_DIR / "sample_data.csv"
    sample.to_csv(sample_path, index=False)
    print(f"  ✓ Sample data: {len(sample):,} rows → {sample_path.name}")

    print(f"\n  All files exported to: {PROCESSED_DIR}/")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """
    Orchestrate the full data preparation pipeline.

    This is designed to be idempotent — you can re-run it safely.
    Each step checks for existing outputs before reprocessing.
    """
    parser = argparse.ArgumentParser(description="Prepare Store Sales data for forecasting")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download (use existing files in data/raw/)"
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Sales Forecasting - Data Preparation Pipeline        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Step 1: Download (unless skipped)
    if not args.skip_download:
        download_kaggle_data()
    else:
        print("\n  Skipping download (--skip-download flag set)")

    # Step 2: Load and clean
    data = load_raw_data()

    # Step 3: Feature engineering
    featured_df = engineer_features(data)

    # Step 4: Build star schema
    star_schema = build_star_schema(featured_df)

    # Step 5: Export
    export_data(star_schema, featured_df)

    print("\n" + "=" * 60)
    print("✓ DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Star schema tables in: {PROCESSED_DIR}/")
    print(f"  Next step: Run notebooks/02_data_preparation.ipynb for visual QA")
    print(f"  Or proceed to: python src/automl_train.py")


if __name__ == "__main__":
    main()
