# System Architecture

## End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                    │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Kaggle   │───→│  data_prep   │───→│ Star Schema  │                  │
│  │  Dataset  │    │    .py       │    │   Tables     │                  │
│  │ (5 CSVs)  │    │              │    │              │                  │
│  └──────────┘    │ • Clean      │    │ • dim_date   │                  │
│                  │ • Merge      │    │ • dim_store  │                  │
│                  │ • Feature    │    │ • dim_product│                  │
│                  │   engineer   │    │ • fact_sales │                  │
│                  └──────────────┘    └──────┬───────┘                  │
│                                             │                          │
└─────────────────────────────────────────────┼──────────────────────────┘
                                              │
                    ┌─────────────────────────┼────────────────────┐
                    │              AZURE ML    │                    │
                    │                         ▼                    │
                    │  ┌──────────────────────────────────────┐    │
                    │  │         AutoML Forecasting            │    │
                    │  │                                       │    │
                    │  │  • 50 model trials (parallel)         │    │
                    │  │  • ARIMA, Prophet, LightGBM, XGBoost  │    │
                    │  │  • TCNForecaster (deep learning)      │    │
                    │  │  • Auto featurization (lags, windows) │    │
                    │  │  • Best model selected by NormRMSE    │    │
                    │  └──────────────┬───────────────────────┘    │
                    │                 │                             │
                    │                 ▼                             │
                    │  ┌──────────────────────────────────────┐    │
                    │  │       Managed Online Endpoint          │    │
                    │  │                                       │    │
                    │  │  • REST API with key authentication   │    │
                    │  │  • Auto-scaling deployment            │    │
                    │  │  • MLflow model serving               │    │
                    │  └──────────────┬───────────────────────┘    │
                    │                 │                             │
                    └─────────────────┼────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────┐
                    │           SCORING PIPELINE                   │
                    │                                              │
                    │  score_forecasts.py                          │
                    │  • Build future date features (90 days)      │
                    │  • Batch score against endpoint              │
                    │  • Post-process predictions                  │
                    │  • Export fact_forecasts table                │
                    └──────────────┬──────────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────────────┐
                    │            POWER BI DASHBOARD                │
                    │                                              │
                    │  ┌─────────────────────────────────────┐    │
                    │  │           Star Schema Model           │    │
                    │  │                                       │    │
                    │  │  dim_date ──┐                         │    │
                    │  │  dim_store ─┼──→ fact_sales            │    │
                    │  │  dim_product┤                         │    │
                    │  │             └──→ fact_forecasts        │    │
                    │  └─────────────────────────────────────┘    │
                    │                                              │
                    │  ┌─────────────────────────────────────┐    │
                    │  │           DAX Measures                │    │
                    │  │                                       │    │
                    │  │  • YoY Growth (SAMEPERIODLASTYEAR)    │    │
                    │  │  • Running Total (TOTALYTD)           │    │
                    │  │  • MAPE Accuracy (SUMX + ABS)         │    │
                    │  │  • % of Total (CALCULATE + ALL)       │    │
                    │  │  • Dynamic Toggle (SWITCH + TRUE)     │    │
                    │  │  • Row-Level Security                 │    │
                    │  └─────────────────────────────────────┘    │
                    └─────────────────────────────────────────────┘
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Source | Kaggle Store Sales (Favorita) | 3M+ transactional records |
| Data Processing | Python (Pandas, NumPy) | ETL and feature engineering |
| ML Platform | Azure Machine Learning (SDK v2) | AutoML training and deployment |
| Compute | AmlCompute (Standard_DS3_v2) | Auto-scaling training cluster |
| Model Serving | Managed Online Endpoint | Real-time prediction API |
| Visualization | Power BI Desktop | Interactive dashboard |
| Data Modeling | DAX + Power Query (M) | Business logic and transforms |
| Version Control | Git + GitHub | Code and documentation |

## Data Flow

```
Kaggle CSVs (5 files)
    │
    ▼
data_prep.py
    │ ── Clean, merge, feature engineer
    │ ── Build star schema dimensions & facts
    │
    ├──→ automl_training_data.parquet (flat, for Azure ML)
    │        │
    │        ▼
    │    automl_train.py → Azure AutoML
    │        │
    │        ▼
    │    deploy_endpoint.py → Online Endpoint
    │        │
    │        ▼
    │    score_forecasts.py → fact_forecasts.csv
    │
    ├──→ dim_date.csv
    ├──→ dim_store.csv        ──→  Power BI Desktop
    ├──→ dim_product.csv              │
    ├──→ fact_sales.csv               ▼
    └──→ fact_forecasts.csv      Dashboard (.pbix)
```

## Azure Resource Architecture

```
Azure Subscription
└── Resource Group: sales-forecasting-rg
    ├── Azure ML Workspace: sales-forecast-ws
    │   ├── Compute: forecast-cluster (0-4 nodes, DS3_v2)
    │   ├── Experiment: store-sales-forecasting
    │   ├── Model: store-sales-forecaster (MLflow)
    │   ├── Endpoint: sales-forecast-endpoint
    │   └── Data: store-sales-training (Parquet)
    ├── Storage Account (auto-created by workspace)
    ├── Key Vault (auto-created, stores secrets)
    └── Application Insights (auto-created, monitoring)
```

## Cost Estimate

| Resource | Cost | Notes |
|----------|------|-------|
| Compute cluster (training) | ~$2-4 | 4 nodes × $0.24/hr × 2hrs |
| Online endpoint (deployed) | ~$5.76/day | 1 node × $0.24/hr × 24hrs |
| Storage | < $0.10 | Small dataset |
| **Total for demo** | **~$8-12** | Delete endpoint after demo! |

**Cost optimization:** Delete the endpoint after generating forecasts.
Re-deploy only when you need to refresh predictions.
