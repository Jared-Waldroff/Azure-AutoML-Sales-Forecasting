# Sales Forecasting Dashboard with Azure AutoML & Power BI

An end-to-end sales forecasting solution that combines Azure Machine Learning's AutoML for predictive modeling with Power BI for interactive visualization. Built on 3M+ daily transactions from the Favorita Store Sales dataset, this project demonstrates the full data science lifecycle: data engineering, ML model training, deployment, and business intelligence.

## The Business Problem

> "Show me how we've performed historically AND where we're headed."

Every retail leadership team needs this. This project delivers it by:
- Forecasting the next 30/60/90 days of sales by store and product category
- Comparing actuals to predictions in a single interactive dashboard
- Providing drill-down by region, product family, and time period

## Architecture

```
Kaggle Dataset → Python ETL → Star Schema → Azure AutoML → Online Endpoint
                                    ↓                            ↓
                               Power BI ← ← ← ← ← fact_forecasts
```

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Processing | Python (Pandas) | ETL, feature engineering, star schema |
| ML Training | Azure AutoML (SDK v2) | Time-series forecasting (50 model trials) |
| Model Serving | Managed Online Endpoint | REST API for real-time predictions |
| Visualization | Power BI Desktop | Interactive dashboard with DAX |
| Data Model | Star Schema | dim_date, dim_store, dim_product → fact tables |

For detailed architecture diagrams, see [docs/architecture.md](docs/architecture.md).

## Dataset

**Kaggle Store Sales — Favorita** ([link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting))

| Metric | Value |
|--------|-------|
| Records | ~3 million |
| Date range | 2013-01-01 to 2017-08-15 |
| Stores | 54 across Ecuador |
| Product families | 33 (grouped into 11 categories) |
| External features | Oil prices, holidays, promotions |

## Azure AutoML Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Primary metric | Normalized RMSE | Scale-independent across product families |
| Forecast horizon | 90 days | Covers 30/60/90 day planning windows |
| Time series ID | store_nbr × family | 1,782 unique series |
| Max trials | 50 | Thorough model search |
| Compute | Standard_DS3_v2 (0→4 nodes) | Cost-optimized auto-scaling |
| Models tried | LightGBM, XGBoost, Prophet, ARIMA, TCN, ElasticNet, etc. | Full spectrum |

### Why the Winner Won

LightGBM typically dominates retail forecasting because it:
- Handles mixed features (categorical stores + numerical prices) natively
- Captures non-linear relationships (promotion diminishing returns)
- Trains 10-100x faster than deep learning, allowing more iterations
- Handles missing values internally

See [docs/model_selection.md](docs/model_selection.md) for the full leaderboard analysis.

### Feature Importance

| Feature | Importance | Business Insight |
|---------|-----------|-----------------|
| Promotions | ~23% | Biggest controllable lever |
| Day of week | ~18% | Strong weekly seasonality |
| Oil price | ~14% | Ecuador's economy tracks oil |
| Month | ~12% | Year-round seasonal patterns |
| Transactions | ~10% | Foot traffic drives volume |

## Power BI Dashboard

### Data Model (Star Schema)

```
dim_date ──────┐
dim_store ─────┼──→ fact_sales (historical actuals)
dim_product ───┤
               └──→ fact_forecasts (ML predictions)
```

### DAX Measures Implemented

| Measure | DAX Pattern | Purpose |
|---------|------------|---------|
| YoY Growth | `SAMEPERIODLASTYEAR` | Year-over-year comparison |
| Running Total | `TOTALYTD` | Year-to-date accumulation |
| MAPE | `SUMX` + `ABS` | Forecast accuracy (actual vs predicted) |
| % of Total | `CALCULATE` + `ALL` | Category contribution analysis |
| Dynamic Toggle | `SWITCH(TRUE(), ...)` | User switches between Revenue/Units/Margin |
| Conditional Colors | `SWITCH(TRUE(), ...)` | Green/red formatting based on growth |

All measures use `VAR/RETURN` for clean, debuggable DAX. Full code and explanations in [powerbi/dax_measures.md](powerbi/dax_measures.md).

### Row-Level Security

Implemented regional RLS so different managers see only their stores:
- **Highland Region**: Andean mountain stores
- **Coastal Region**: Pacific coast stores
- **Amazon Region**: Amazon basin stores

Setup guide: [powerbi/rls_setup.md](powerbi/rls_setup.md).

## Project Structure

```
├── data/
│   ├── raw/                    # Kaggle CSVs (not in git)
│   ├── processed/              # Star schema tables
│   └── output/                 # Forecasts and evaluation results
├── notebooks/
│   ├── 01_data_exploration     # EDA and data quality
│   ├── 02_data_preparation     # ETL pipeline walkthrough
│   ├── 03_automl_training      # Azure AutoML experiment
│   ├── 04_model_evaluation     # Leaderboard and metrics analysis
│   └── 05_endpoint_scoring     # Deployment and forecast generation
├── src/
│   ├── data_prep.py            # Data download, clean, star schema
│   ├── automl_train.py         # AutoML configuration and submission
│   ├── model_evaluate.py       # Metrics, charts, feature importance
│   ├── deploy_endpoint.py      # Model deployment to endpoint
│   └── score_forecasts.py      # Score 90-day predictions
├── powerbi/
│   ├── data_model.md           # Star schema import guide
│   ├── dax_measures.md         # All DAX with explanations
│   ├── power_query.md          # M code transformations
│   └── rls_setup.md            # Row-Level Security setup
├── docs/
│   ├── architecture.md         # System architecture diagrams
│   ├── model_selection.md      # AutoML leaderboard analysis
│   └── employer_talking_points.md
├── screenshots/                # Dashboard and leaderboard images
├── .env.template               # Azure config (copy to .env)
├── requirements.txt            # Python dependencies
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.9+
- Azure subscription with ML workspace
- Power BI Desktop (Windows)
- Kaggle account (for data download)

### Setup

```bash
# 1. Clone and setup
git clone <repo-url>
cd Azure-AutoML
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure Azure credentials
cp .env.template .env
# Edit .env with your Azure subscription details

# 3. Authenticate with Azure
az login

# 4. Download and prepare data
python src/data_prep.py

# 5. Train the model (takes ~2 hours)
python src/automl_train.py

# 6. Evaluate results
python src/model_evaluate.py --job-name <job-name-from-step-5>

# 7. Deploy and score
python src/deploy_endpoint.py --job-name <job-name-from-step-5>
python src/score_forecasts.py

# 8. Import into Power BI
# Open Power BI Desktop → follow powerbi/data_model.md
```

### Demo Mode (no Azure needed)

```bash
# Generate synthetic forecasts for Power BI development
python src/data_prep.py --skip-download  # Requires Kaggle data in data/raw/
python src/model_evaluate.py --demo       # Generate sample charts
python src/score_forecasts.py --demo      # Generate synthetic forecasts
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Azure ML SDK version | v2 (azure-ai-ml) | Modern, recommended SDK |
| Data format | Parquet + CSV | Parquet for Python (fast), CSV for Power BI |
| Star schema | Separate fact tables | Actuals and forecasts share dimensions |
| Compute | Auto-scaling cluster | Scales to 0 when idle (cost optimization) |
| Metric | Normalized RMSE | Fair comparison across different-scale series |
| RLS | Region-based | Demonstrates enterprise data governance |

## Cost

| Resource | Cost |
|----------|------|
| AutoML training (~2 hours) | ~$2-4 |
| Endpoint (while deployed) | ~$5.76/day |
| Storage | < $0.10 |
| **Total for demo** | **~$8-12** |

**Delete the endpoint after generating forecasts to stop charges.**

## Documentation

- [Architecture](docs/architecture.md) — Full pipeline and Azure resource diagrams
- [Model Selection](docs/model_selection.md) — AutoML leaderboard and winner analysis
- [DAX Measures](powerbi/dax_measures.md) — All DAX code with explanations
- [Data Model](powerbi/data_model.md) — Star schema setup in Power BI
- [Power Query](powerbi/power_query.md) — M transformations for each table
- [Row-Level Security](powerbi/rls_setup.md) — RLS configuration guide

## Author

Jared Waldroff
