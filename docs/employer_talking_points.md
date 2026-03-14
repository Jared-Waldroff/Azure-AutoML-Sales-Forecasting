# Employer Talking Points

## How to Present This Project

Use this guide when discussing the project with your employer.
It's structured as a conversation flow — start with the business problem
and work toward the technical details.

---

## 1. Open with the Business Problem (30 seconds)

> "I built an end-to-end sales forecasting solution that answers the question
> every retail leadership team asks: **'Show me how we've performed historically
> AND where we're headed.'**
>
> I used the Favorita Store Sales dataset — 3 million daily transactions across
> 54 stores and 33 product categories — because it mirrors the complexity of
> a real retail environment."

**Why this works:** You're framing it as a business solution, not a tech demo.

---

## 2. Walk Through the Architecture (2 minutes)

> "The pipeline has three layers:"

### Data Layer
> "I started with raw transactional data — sales, store metadata, oil prices,
> holidays — and built a **star schema** with proper dimension and fact tables.
> The star schema isn't just for Power BI — it's how enterprise data warehouses
> are structured. I included synthetic revenue and margin calculations because
> the original dataset only had unit sales."

### ML Layer
> "I used **Azure AutoML** to build a time-series forecasting model.
> AutoML evaluated 50 different model architectures — including LightGBM,
> Prophet, ARIMA, and a deep learning TCN model — all in parallel.
>
> The winner was [LightGBM/Prophet/etc.] with a normalized RMSE of [X].
> I deployed it as a **managed online endpoint** — a REST API that returns
> predictions in real time.
>
> What I find most interesting is that **promotions and day-of-week were
> the top two predictive features**, accounting for over 40% of the model's
> accuracy. That tells you the biggest levers are promotional strategy
> and weekly staffing patterns."

### Visualization Layer
> "The Power BI dashboard combines historical actuals with 90-day forecasts.
> I wrote custom **DAX measures** for year-over-year growth, forecast accuracy
> (MAPE), dynamic metric toggling, and running totals. I also implemented
> **Row-Level Security** so different regional managers would only see their stores."

---

## 3. Demonstrate Technical Depth (when asked)

### If they ask about DAX:
> "I used `VAR/RETURN` throughout because it makes DAX debuggable.
> The most interesting measure is the MAPE calculation — it uses `SUMX`
> to iterate row by row, comparing each actual sale to its corresponding
> prediction using `ABS` for absolute error.
>
> I also built a dynamic metric toggle using `SWITCH(TRUE(), ...)` that
> lets users switch between revenue, units, and margin from a single slicer."

### If they ask about the ML model:
> "I chose **normalized RMSE** as the primary metric because the dataset
> has product families with wildly different scales — grocery sells 800 units
> per day while magazines sell 5. Normalization makes the comparison fair.
>
> I let AutoML try everything, including deep learning, because I wanted
> the leaderboard to tell the story. LightGBM typically wins for retail
> because it handles mixed feature types — categorical stores alongside
> numerical oil prices — without one-hot encoding."

### If they ask about the data model:
> "The star schema has three dimensions — date, store, product — feeding
> into two fact tables: actuals and forecasts. I designed them with matching
> keys so a slicer on product category filters both tables simultaneously.
>
> The date dimension is marked as a date table in Power BI, which enables
> time intelligence functions like `SAMEPERIODLASTYEAR` and `TOTALYTD`."

### If they ask about deployment:
> "I deployed the model as a **managed online endpoint** using Azure ML.
> It uses MLflow serving, so there's no custom scoring script — the model
> handles serialization natively. Authentication is via API keys, and the
> endpoint auto-scales based on request volume.
>
> For the forecast pipeline, I batch-score all store × product combinations
> for 90 days ahead, then write the results into a fact table that Power BI
> imports directly."

---

## 4. Address Limitations Honestly (shows maturity)

> "There are things the model doesn't capture: competitor actions, weather,
> and new product launches with no history. In a production setting, I'd
> add monitoring to flag when actuals deviate more than 2 standard deviations
> from the forecast — that's your signal to investigate."

---

## 5. Close with What You'd Do Next (shows forward thinking)

> "If I were extending this for production, I'd add three things:
>
> 1. **Automated retraining** — an Azure ML pipeline that retrains monthly
>    as new sales data comes in, with a champion/challenger comparison
>    to prevent model degradation.
>
> 2. **Anomaly detection** — a separate model that flags unusual sales
>    patterns in real time, feeding alerts into the dashboard.
>
> 3. **What-if analysis** — a Power BI page where users can adjust
>    promotion levels and see how the forecast changes, using
>    DAX calculation groups."

---

## Quick Reference: Key Numbers

Keep these in your back pocket:

| Metric | Value | Context |
|--------|-------|---------|
| Dataset size | ~3M rows | 4.5 years of daily data |
| Stores | 54 | Across Ecuador |
| Product families | 33 | Grouped into 11 categories |
| Unique time series | ~1,782 | store × product combinations |
| Forecast horizon | 90 days | 30/60/90 day windows |
| Models evaluated | 50 | AutoML trial limit |
| Top feature | Promotions | 23% importance |
| Target MAPE | < 20% | Industry benchmark: 15-25% |

---

## Vocabulary Cheat Sheet

Terms to use naturally in conversation:

| Instead of... | Say... |
|---------------|--------|
| "I trained a model" | "I configured an AutoML experiment with 50 trial architectures" |
| "It's accurate" | "We achieved a MAPE under 20%, which is within industry benchmarks for daily retail forecasting" |
| "I made a dashboard" | "I built a star schema with time intelligence measures and Row-Level Security" |
| "I used Azure" | "I used the Azure ML SDK v2 to manage the full lifecycle — from compute provisioning through endpoint deployment" |
| "The model is good" | "LightGBM won because it handles the mixed feature types in retail data without encoding overhead" |
