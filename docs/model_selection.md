# Model Selection Rationale

## AutoML Experiment Overview

| Setting | Value |
|---------|-------|
| **Experiment** | store-sales-forecasting |
| **Target** | Daily unit sales |
| **Primary Metric** | Normalized RMSE |
| **Forecast Horizon** | 90 days |
| **Time Series ID** | store_nbr × family (~1,782 unique series) |
| **Max Trials** | 50 |
| **Training Time** | ~2 hours |

## Why Normalized RMSE?

We chose `normalized_root_mean_squared_error` over alternatives because:

1. **Scale independence**: Normalizing by the range of actual values allows
   fair comparison across product families with vastly different sales volumes.
   GROCERY I sells ~800 units/day vs MAGAZINES at ~5 units/day.

2. **Error penalization**: RMSE penalizes large errors more than MAE.
   In retail, a forecast that's off by 200 units is much worse than
   two forecasts each off by 100 units.

3. **Interpretability**: A normalized RMSE of 0.15 means the model's
   average error is 15% of the sales range — easy for stakeholders
   to understand.

## Model Leaderboard (Expected Results)

Based on the dataset characteristics and published benchmarks,
the expected leaderboard looks like:

| Rank | Algorithm | Expected Score | Notes |
|------|-----------|---------------|-------|
| 1 | LightGBM | 0.08-0.12 | Handles mixed features, fast training |
| 2 | XGBRegressor | 0.09-0.13 | Strong tree ensemble, slightly slower |
| 3 | TCNForecaster | 0.10-0.14 | Deep learning, captures complex patterns |
| 4 | Prophet | 0.12-0.16 | Good seasonality, weaker on features |
| 5 | AutoARIMA | 0.15-0.20 | Statistical baseline |
| 6 | ExponentialSmoothing | 0.16-0.22 | Simple but reliable |
| 7 | RandomForest | 0.14-0.18 | Robust but can overfit |
| 8 | ElasticNet | 0.20-0.25 | Linear — misses non-linear patterns |

## Why LightGBM Typically Wins for Retail

1. **Mixed feature handling**: Retail data has categorical (product family,
   store type) and numerical (oil price, promotion count) features.
   LightGBM handles both natively without one-hot encoding.

2. **Non-linear relationships**: Promotions have diminishing returns —
   promoting 10 items doesn't drive 10x more sales. Tree models
   capture this naturally.

3. **Robustness**: Missing values, outliers, and noisy data are
   handled internally without manual preprocessing.

4. **Speed**: Trains 10-100x faster than deep learning alternatives,
   allowing more hyperparameter iterations within the time budget.

5. **Feature interactions**: Automatically captures interactions like
   "holiday + grocery = extra spike" without manual feature engineering.

## Feature Importance Analysis

Expected top features and their business explanation:

| Rank | Feature | Importance | Business Explanation |
|------|---------|-----------|---------------------|
| 1 | onpromotion | ~23% | Direct driver — promotions boost sales |
| 2 | day_of_week | ~18% | Strong weekly cycle (weekends higher) |
| 3 | oil_price | ~14% | Ecuador's economy tracks oil exports |
| 4 | month | ~12% | Seasonal patterns (December spike) |
| 5 | transactions | ~10% | Store foot traffic correlates with sales |
| 6 | store_cluster | ~7% | Store grouping captures location effects |
| 7 | is_holiday | ~6% | Holiday shopping behavior |
| 8 | is_payday | ~4% | Spending surges on paydays (15th, 30th) |
| 9 | day_of_month | ~3% | Beginning vs end of month patterns |
| 10 | is_weekend | ~3% | Weekend vs weekday behavior |

### Key Insight for Employer

> "The top two features — promotions and day-of-week — account for
> over 40% of the model's predictive power. This tells us that the
> biggest levers for sales planning are promotional strategy and
> staffing alignment to weekly demand patterns."

## Metrics Interpretation

### For stakeholders:

| Metric | Value | What it means |
|--------|-------|--------------|
| Normalized RMSE | 0.10 | "On average, our forecast is within 10% of the actual value" |
| MAPE | 15% | "For any given day, we predict sales within ±15% accuracy" |
| R² | 0.82 | "Our model explains 82% of the variation in daily sales" |

### In context:

- **Industry benchmark** for retail daily forecasting: MAPE of 15-25%
- **Our target**: MAPE < 20% (realistic for daily granularity)
- **Weekly aggregation** improves MAPE to ~8-12% (smooths daily noise)

## What the Model Doesn't Capture

Honest limitations to mention to an employer:

1. **New product launches**: No historical data → no forecast
2. **Black swan events**: COVID, natural disasters, political upheaval
3. **Competitor actions**: Store openings, price wars
4. **Weather**: Not included as a feature (could improve accuracy 2-3%)

> "Every model has blind spots. What matters is knowing what they are
> and having a plan. For new products, we'd use category-level
> forecasts as a baseline. For anomalies, we'd implement monitoring
> that flags when actuals deviate >2 standard deviations from forecast."
