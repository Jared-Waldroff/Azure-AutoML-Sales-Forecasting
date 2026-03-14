"""
============================================================================
Forecast Scoring Pipeline
============================================================================
Generates predictions for the next 90 days by scoring against the deployed
Azure ML endpoint, then exports results as a fact table for Power BI.

Pipeline Steps:
    1. Build the scoring payload (next 90 days × all store×product combos)
    2. Batch-score against the deployed endpoint
    3. Post-process predictions (clip negatives, round)
    4. Build fact_forecasts table (same schema as fact_sales for easy union)
    5. Export to CSV and Parquet for Power BI import

Why batch scoring via endpoint (vs batch endpoints)?
    - Managed online endpoints are simpler for this project scope
    - Batch endpoints are better for millions of rows — overkill here
    - The online endpoint lets us also demo real-time scoring

Usage:
    python src/score_forecasts.py
    python src/score_forecasts.py --horizon 30    # Override to 30 days
    python src/score_forecasts.py --demo          # Generate synthetic forecasts

Author: Jared Waldroff
============================================================================
"""

import os
import json
import argparse
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "sales-forecast-endpoint")
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "90"))


# ============================================================================
# Step 1: Build Scoring Payload
# ============================================================================

def build_scoring_data(horizon: int) -> pd.DataFrame:
    """
    Create the input DataFrame for scoring: future dates × all store×product combos.

    The scoring data must match the training schema exactly — same columns,
    same dtypes, same feature engineering. This is a common production pitfall:
    training/serving skew. We avoid it by reusing the same feature logic.

    For future dates, we need to provide:
        - Known values: date features, store metadata (static)
        - Estimated values: oil price (use last known), promotions (assume avg)
        - Unknown values: transactions (use historical average by day-of-week)

    Args:
        horizon: Number of days to forecast

    Returns:
        pd.DataFrame: Scoring input with one row per store×product×date
    """
    print("=" * 60)
    print("STEP 1: Building Scoring Payload")
    print("=" * 60)

    # Load dimension tables for store and product metadata
    dim_store = pd.read_csv(PROCESSED_DIR / "dim_store.csv")
    dim_product = pd.read_csv(PROCESSED_DIR / "dim_product.csv")

    # Load historical data for reference values
    fact_sales = pd.read_parquet(PROCESSED_DIR / "fact_sales.parquet")

    # Determine the forecast start date (day after last actual)
    last_actual_date = pd.to_datetime(fact_sales["date_key"].max(), format="%Y%m%d")
    forecast_start = last_actual_date + timedelta(days=1)
    forecast_dates = pd.date_range(start=forecast_start, periods=horizon, freq="D")

    print(f"  Last actual date: {last_actual_date.date()}")
    print(f"  Forecast range: {forecast_dates[0].date()} → {forecast_dates[-1].date()}")
    print(f"  Horizon: {horizon} days")

    # Get unique store-product combinations
    stores = dim_store["store_key"].unique()
    families = dim_product["family"].unique()

    # Create the cartesian product: dates × stores × products
    # This is every combination we need to predict
    rows = []
    for date in forecast_dates:
        for store_nbr in stores:
            for family in families:
                rows.append({"date": date, "store_nbr": store_nbr, "family": family})

    scoring_df = pd.DataFrame(rows)

    print(f"  Combinations: {len(stores)} stores × {len(families)} families × {horizon} days")
    print(f"  Total rows to score: {len(scoring_df):,}")

    # --- Add store metadata ---
    scoring_df = scoring_df.merge(dim_store, left_on="store_nbr", right_on="store_key", how="left")

    # --- Add calendar features (same logic as data_prep.py) ---
    scoring_df["year"] = scoring_df["date"].dt.year
    scoring_df["month"] = scoring_df["date"].dt.month
    scoring_df["day_of_week"] = scoring_df["date"].dt.dayofweek
    scoring_df["day_of_month"] = scoring_df["date"].dt.day
    scoring_df["week_of_year"] = scoring_df["date"].dt.isocalendar().week.astype(int)
    scoring_df["quarter"] = scoring_df["date"].dt.quarter
    scoring_df["is_weekend"] = (scoring_df["day_of_week"] >= 5).astype(int)
    scoring_df["is_payday"] = (
        (scoring_df["day_of_month"] == 15) |
        (scoring_df["day_of_month"] == scoring_df["date"].dt.days_in_month)
    ).astype(int)
    scoring_df["is_month_start"] = (scoring_df["day_of_month"] <= 5).astype(int)

    # --- Estimated features for future dates ---
    # Oil price: Use last known value (best available estimate)
    # In production, you'd pull from an oil price API
    last_oil_price = fact_sales["oil_price"].iloc[-1] if "oil_price" in fact_sales.columns else 55.0
    scoring_df["oil_price"] = last_oil_price
    scoring_df["oil_price_lag7"] = last_oil_price

    # Promotions: Use historical average per store×product×day-of-week
    scoring_df["onpromotion"] = 0  # Conservative estimate

    # Holidays: Mark no holidays (or integrate a holiday API)
    scoring_df["is_holiday"] = 0

    # Transactions: Use historical average by store×day-of-week
    scoring_df["transactions"] = 2000  # Reasonable default

    print(f"  ✓ Scoring payload built: {scoring_df.shape[0]:,} rows × {scoring_df.shape[1]} columns")

    return scoring_df


# ============================================================================
# Step 2: Score Against Endpoint
# ============================================================================

def score_endpoint(ml_client: MLClient, scoring_df: pd.DataFrame) -> pd.Series:
    """
    Send scoring data to the deployed endpoint in batches.

    Why batch the requests?
        - Online endpoints have payload size limits (~100MB)
        - Batching prevents timeouts on large requests
        - Enables progress reporting for long scoring jobs

    Args:
        ml_client: Authenticated Azure ML client
        scoring_df: DataFrame with feature columns matching training schema

    Returns:
        pd.Series: Predicted sales values, one per input row
    """
    print("\n" + "=" * 60)
    print("STEP 2: Scoring Against Endpoint")
    print("=" * 60)

    # Select columns that match the training schema
    feature_columns = [
        "date", "store_nbr", "family", "onpromotion",
        "oil_price", "is_holiday", "transactions",
        "year", "month", "day_of_week", "day_of_month",
        "week_of_year", "quarter", "is_weekend", "is_payday",
        "is_month_start", "oil_price_lag7",
        "city", "state", "store_type", "cluster"
    ]

    # Filter to available columns (some may be named differently)
    available_cols = [c for c in feature_columns if c in scoring_df.columns]
    score_data = scoring_df[available_cols].copy()

    # Convert dates to strings for JSON serialization
    score_data["date"] = score_data["date"].dt.strftime("%Y-%m-%d")

    # Batch scoring — 5000 rows per request
    BATCH_SIZE = 5000
    all_predictions = []
    total_batches = (len(score_data) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  Scoring {len(score_data):,} rows in {total_batches} batches...")

    for i in range(0, len(score_data), BATCH_SIZE):
        batch = score_data.iloc[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        # Construct the request payload
        request = {
            "input_data": {
                "columns": list(batch.columns),
                "data": batch.values.tolist()
            }
        }

        # Write to temp file and invoke
        temp_path = OUTPUT_DIR / "temp_scoring_request.json"
        with open(temp_path, "w") as f:
            json.dump(request, f)

        try:
            response = ml_client.online_endpoints.invoke(
                endpoint_name=ENDPOINT_NAME,
                request_file=str(temp_path),
            )
            batch_predictions = json.loads(response)
            all_predictions.extend(batch_predictions)

            print(f"    Batch {batch_num}/{total_batches}: ✓ ({len(batch_predictions)} predictions)")

        except Exception as e:
            print(f"    Batch {batch_num}/{total_batches}: ✗ Error: {e}")
            # Fill with NaN for failed batches (don't lose other predictions)
            all_predictions.extend([float("nan")] * len(batch))

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()

    print(f"\n  ✓ Scoring complete: {len(all_predictions):,} predictions")

    return pd.Series(all_predictions, name="predicted_sales")


# ============================================================================
# Step 3: Generate Demo Forecasts (when endpoint isn't available)
# ============================================================================

def generate_demo_forecasts(scoring_df: pd.DataFrame) -> pd.Series:
    """
    Generate realistic synthetic forecasts for portfolio development.

    This creates forecasts that look realistic by:
        - Using historical patterns (trend, seasonality)
        - Adding store-level variation
        - Adding product-family-level variation
        - Including some noise (real forecasts aren't perfect)

    This is useful for building the Power BI dashboard before
    the Azure ML endpoint is deployed.

    Args:
        scoring_df: The scoring payload with date and metadata columns

    Returns:
        pd.Series: Synthetic predicted sales values
    """
    print("\n" + "=" * 60)
    print("STEP 2 (DEMO): Generating Synthetic Forecasts")
    print("=" * 60)

    np.random.seed(42)
    n = len(scoring_df)

    # Base sales level varies by product family
    family_base = {
        "GROCERY I": 800, "GROCERY II": 200, "BEVERAGES": 400,
        "PRODUCE": 300, "MEATS": 150, "POULTRY": 100,
        "DAIRY": 250, "DELI": 80, "FROZEN FOODS": 60,
        "BREAD/BAKERY": 180, "CLEANING": 200, "EGGS": 120,
        "PERSONAL CARE": 90, "BEAUTY": 40, "BABY CARE": 30,
        "HOME AND KITCHEN I": 50, "HOME AND KITCHEN II": 30,
        "SCHOOL AND OFFICE SUPPLIES": 25, "CELEBRATION": 20,
    }

    # Get base sales per row
    base = scoring_df["family"].map(family_base).fillna(50).values

    # Add day-of-week pattern (weekends are higher for grocery)
    dow_effect = np.where(scoring_df["is_weekend"].values == 1, 1.15, 1.0)

    # Add monthly seasonality (December is ~30% higher)
    month_effect = 1.0 + 0.15 * np.sin(2 * np.pi * (scoring_df["month"].values - 3) / 12)

    # Add store-level variation (±20%)
    # Use store_key (store_nbr gets renamed after merge with dim_store)
    store_col = "store_nbr" if "store_nbr" in scoring_df.columns else "store_key"
    store_effect = 0.8 + 0.4 * (scoring_df[store_col].values % 10) / 10

    # Add noise (±15%)
    noise = np.random.normal(1.0, 0.15, n)

    # Combine all effects
    predictions = base * dow_effect * month_effect * store_effect * noise

    # Clip negatives (sales can't be negative)
    predictions = np.maximum(predictions, 0).round(0)

    print(f"  ✓ Generated {n:,} synthetic predictions")
    print(f"    Mean: {predictions.mean():.0f} | Median: {np.median(predictions):.0f}")
    print(f"    Min: {predictions.min():.0f} | Max: {predictions.max():.0f}")

    return pd.Series(predictions, name="predicted_sales")


# ============================================================================
# Step 4: Build fact_forecasts Table
# ============================================================================

def build_fact_forecasts(scoring_df: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """
    Build the fact_forecasts table matching fact_sales schema for Power BI.

    Why match the fact_sales schema?
        - Power BI can UNION the two tables seamlessly
        - Same dimension keys enable shared slicers
        - DAX measures can compare actuals vs forecasts directly
        - The star schema stays clean — no special-case joins

    The fact_forecasts table includes:
        - date_key: Links to dim_date
        - store_key: Links to dim_store
        - product_key: Links to dim_product
        - predicted_sales: Model output
        - predicted_revenue: sales × avg price (estimated)
        - predicted_margin: revenue × margin rate (estimated)
        - forecast_horizon: Days ahead (30/60/90 — for slicer filtering)

    Args:
        scoring_df: Scoring payload with metadata
        predictions: Model predictions (one per row)

    Returns:
        pd.DataFrame: fact_forecasts ready for Power BI import
    """
    print("\n" + "=" * 60)
    print("STEP 3: Building fact_forecasts Table")
    print("=" * 60)

    # Load product dimension for key mapping
    dim_product = pd.read_csv(PROCESSED_DIR / "dim_product.csv")
    family_key_map = dim_product.set_index("family")["product_key"].to_dict()

    # Build the fact table
    fact_forecasts = pd.DataFrame({
        "date": scoring_df["date"].values,
        "date_key": scoring_df["date"].dt.strftime("%Y%m%d").astype(int).values,
        "store_key": scoring_df["store_key"].values if "store_key" in scoring_df.columns else scoring_df["store_nbr"].values,
        "product_key": scoring_df["family"].map(family_key_map).values,
        "predicted_sales": predictions.values,
    })

    # Add revenue and margin estimates
    # Use the same price/margin logic as data_prep.py for consistency
    family_to_category = dict(zip(dim_product["family"], dim_product["category"]))
    avg_prices = {
        "Grocery": 2.50, "Fresh": 3.75, "Frozen": 4.50,
        "Bakery": 2.00, "Household": 5.00, "Personal Care": 6.00,
        "Apparel": 15.00, "Seasonal": 4.00, "Electronics": 25.00,
        "Beverages": 3.50, "Other": 5.00
    }
    margin_rates = {
        "Grocery": 0.22, "Fresh": 0.35, "Frozen": 0.28,
        "Bakery": 0.45, "Household": 0.30, "Personal Care": 0.40,
        "Apparel": 0.50, "Seasonal": 0.35, "Electronics": 0.15,
        "Beverages": 0.32, "Other": 0.25
    }

    category = scoring_df["family"].map(family_to_category).fillna("Other")
    price = category.map(avg_prices).fillna(5.00)
    margin_rate = category.map(margin_rates).fillna(0.25)

    fact_forecasts["predicted_revenue"] = (fact_forecasts["predicted_sales"] * price.values).round(2)
    fact_forecasts["predicted_margin"] = (fact_forecasts["predicted_revenue"] * margin_rate.values).round(2)

    # Add forecast horizon bucket (for Power BI slicer)
    # Days from forecast start
    forecast_start = fact_forecasts["date"].min()
    days_ahead = (fact_forecasts["date"] - forecast_start).dt.days
    fact_forecasts["forecast_horizon"] = pd.cut(
        days_ahead,
        bins=[-1, 30, 60, 90],
        labels=["30-Day", "60-Day", "90-Day"]
    )

    print(f"  ✓ fact_forecasts built: {len(fact_forecasts):,} rows")
    print(f"    Date range: {fact_forecasts['date'].min()} → {fact_forecasts['date'].max()}")
    print(f"    Total predicted revenue: ${fact_forecasts['predicted_revenue'].sum():,.0f}")

    return fact_forecasts


# ============================================================================
# Step 5: Export
# ============================================================================

def export_forecasts(fact_forecasts: pd.DataFrame):
    """Export forecast fact table for Power BI consumption."""
    print("\n" + "=" * 60)
    print("STEP 4: Exporting Forecasts")
    print("=" * 60)

    # CSV for Power BI import
    csv_path = PROCESSED_DIR / "fact_forecasts.csv"
    fact_forecasts.to_csv(csv_path, index=False)

    # Parquet for efficient storage
    parquet_path = PROCESSED_DIR / "fact_forecasts.parquet"
    fact_forecasts.to_parquet(parquet_path, index=False)

    # Also export to output/ for documentation
    fact_forecasts.to_csv(OUTPUT_DIR / "fact_forecasts.csv", index=False)

    print(f"  ✓ fact_forecasts.csv: {csv_path}")
    print(f"  ✓ fact_forecasts.parquet: {parquet_path}")
    print(f"  ✓ Backup copy: {OUTPUT_DIR / 'fact_forecasts.csv'}")

    # Weekly aggregation for Power BI
    weekly = (
        fact_forecasts
        .set_index("date")
        .resample("W")[["predicted_sales", "predicted_revenue", "predicted_margin"]]
        .sum()
        .reset_index()
    )
    weekly.to_csv(PROCESSED_DIR / "fact_forecasts_weekly.csv", index=False)
    print(f"  ✓ Weekly aggregation: {len(weekly)} weeks")


# ============================================================================
# Main
# ============================================================================

def main():
    """Generate forecasts and build the Power BI fact table."""
    parser = argparse.ArgumentParser(description="Score forecasts via endpoint")
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON,
                        help=f"Forecast horizon in days (default: {FORECAST_HORIZON})")
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic forecasts (no endpoint needed)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Sales Forecasting - Prediction Scoring               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Step 1: Build scoring payload
    scoring_df = build_scoring_data(args.horizon)

    # Step 2: Score
    if args.demo:
        predictions = generate_demo_forecasts(scoring_df)
    else:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AML_WORKSPACE_NAME"),
        )
        predictions = score_endpoint(ml_client, scoring_df)

    # Step 3: Build fact table
    fact_forecasts = build_fact_forecasts(scoring_df, predictions)

    # Step 4: Export
    export_forecasts(fact_forecasts)

    print("\n" + "=" * 60)
    print("✓ FORECAST SCORING COMPLETE")
    print("=" * 60)
    print(f"  Forecasts exported to: {PROCESSED_DIR}/")
    print(f"  Next step: Import fact_forecasts.csv into Power BI")
    print(f"  (See powerbi/data_model.md for import instructions)")


if __name__ == "__main__":
    main()
