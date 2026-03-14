# Power BI Data Model — Star Schema Setup Guide

## Overview

This guide walks you through importing the star schema tables into Power BI
Desktop and configuring the relationships. The model is designed for optimal
Vertipaq compression and fast DAX calculations.

## Step 1: Import Data

In Power BI Desktop:
1. **Get Data** → **Text/CSV**
2. Import these files from `data/processed/`:

| File | Table Name | Type |
|------|-----------|------|
| `dim_date.csv` | dim_date | Dimension |
| `dim_store.csv` | dim_store | Dimension |
| `dim_product.csv` | dim_product | Dimension |
| `fact_sales.csv` | fact_sales | Fact |
| `fact_forecasts.csv` | fact_forecasts | Fact |

**Tip:** Use "Transform Data" to verify column types before loading.

## Step 2: Configure Relationships

Create these relationships in the Model view (drag key → key):

```
dim_date.date_key ──(1:*)──→ fact_sales.date_key
dim_date.date_key ──(1:*)──→ fact_forecasts.date_key
dim_store.store_key ──(1:*)──→ fact_sales.store_key
dim_store.store_key ──(1:*)──→ fact_forecasts.store_key
dim_product.product_key ──(1:*)──→ fact_sales.product_key
dim_product.product_key ──(1:*)──→ fact_forecasts.product_key
```

### Relationship Properties
- **Cardinality:** One-to-Many (dimension → fact)
- **Cross-filter direction:** Single (dimension filters fact)
- **Active:** All relationships active

### Why these relationships matter:
- Slicers on dimensions (date, store, product) automatically filter both
  fact tables — so users can compare actuals vs forecasts with the same filters
- Single-direction cross-filtering prevents ambiguous paths
- The date dimension enables all DAX time intelligence functions

## Step 3: Mark the Date Table

**Critical for time intelligence DAX to work:**

1. Select `dim_date` in the Model view
2. Go to **Table tools** → **Mark as date table**
3. Select the `date` column as the date column

Without this, `SAMEPERIODLASTYEAR`, `TOTALYTD`, and other time intelligence
functions will not work correctly.

## Step 4: Configure Column Properties

### dim_date
| Column | Data Category | Format |
|--------|--------------|--------|
| date | Date | yyyy-mm-dd |
| year | - | Whole number |
| month_name | - | Text |
| day_name | - | Text |

### dim_store
| Column | Data Category | Format |
|--------|--------------|--------|
| city | City | Text |
| state | State/Province | Text |
| region | - | Text |

### fact_sales / fact_forecasts
| Column | Format | Summarization |
|--------|--------|---------------|
| sales/predicted_sales | Whole number | Sum |
| revenue/predicted_revenue | Currency ($) | Sum |
| margin/predicted_margin | Currency ($) | Sum |

## Step 5: Create Measure Table

Best practice: Create a dedicated table for DAX measures.

1. **Modeling** → **New Table**
2. Enter: `_Measures = { BLANK() }`
3. All measures from `dax_measures.md` go in this table

Why a measure table?
- Keeps measures organized and separate from data columns
- Users can easily find all calculated metrics in one place
- Convention used in professional Power BI deployments

## Star Schema Diagram

```
                    ┌─────────────────────────┐
                    │       dim_date           │
                    ├─────────────────────────┤
                    │ date_key (PK)           │
                    │ date                    │
                    │ year, month, quarter    │
                    │ month_name, day_name    │
                    │ week_of_year            │
                    │ is_weekend, is_holiday  │
                    │ year_month              │
                    └──────────┬──────────────┘
                               │ 1:*
            ┌──────────────────┼──────────────────┐
            │                  │                   │
            ▼                  ▼                   ▼
┌───────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│    dim_store      │ │   fact_sales     │ │   fact_forecasts     │
├───────────────────┤ ├──────────────────┤ ├──────────────────────┤
│ store_key (PK)    │ │ date_key (FK)    │ │ date_key (FK)        │
│ store_nbr         │ │ store_key (FK)   │ │ store_key (FK)       │
│ city, state       │ │ product_key (FK) │ │ product_key (FK)     │
│ store_type        │ │ sales            │ │ predicted_sales      │
│ cluster           │ │ revenue          │ │ predicted_revenue    │
│ region (for RLS)  │ │ margin           │ │ predicted_margin     │
└───────┬───────────┘ │ onpromotion      │ │ forecast_horizon     │
        │ 1:*         │ oil_price        │ └──────────┬───────────┘
        └─────────────┤ is_holiday       │            │
                      │ transactions     │            │
                      └──────────────────┘            │
                               ▲                      │
                               │ 1:*                  │
                    ┌──────────┴──────────┐           │
                    │    dim_product      │───────────┘
                    ├─────────────────────┤    1:*
                    │ product_key (PK)    │
                    │ family              │
                    │ category            │
                    └─────────────────────┘
```

## Step 6: Verify

After setup, verify by creating a simple matrix:
- Rows: `dim_product[family]`
- Values: `SUM(fact_sales[revenue])`, `SUM(fact_forecasts[predicted_revenue])`
- Filter: `dim_date[year]` = 2017

If both columns show data, your model is working correctly.

## Next Steps

- Apply DAX measures from `powerbi/dax_measures.md`
- Configure Power Query transforms from `powerbi/power_query.md`
- Set up Row-Level Security from `powerbi/rls_setup.md`
