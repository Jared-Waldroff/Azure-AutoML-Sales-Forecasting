# DAX Measures — Complete Reference

## Overview

All measures use the `VAR/RETURN` pattern for readability and performance.
Create these in the `_Measures` table (see `data_model.md` Step 5).

Each measure includes:
- The DAX code
- What it does
- Why it's written this way
- When to use it

---

## 1. Core Revenue Measure

```dax
Total Revenue =
// Base revenue measure — the foundation for all revenue calculations.
// Explicitly using SUM makes the aggregation clear to anyone reading this.
VAR _TotalRevenue = SUM(fact_sales[revenue])
RETURN
    _TotalRevenue
```

```dax
Total Units =
// Total units sold — used in the dynamic measure toggle
VAR _TotalUnits = SUM(fact_sales[sales])
RETURN
    _TotalUnits
```

```dax
Total Margin =
// Gross margin = Revenue - COGS. Pre-calculated in the fact table.
VAR _TotalMargin = SUM(fact_sales[margin])
RETURN
    _TotalMargin
```

---

## 2. Year-over-Year Growth (SAMEPERIODLASTYEAR)

```dax
Revenue YoY Growth % =
// Compare current period revenue to the same period last year.
// SAMEPERIODLASTYEAR shifts the date filter context back 12 months.
//
// Example: If a slicer shows Jan 2017, this compares to Jan 2016.
// This is the #1 KPI executives ask for in retail dashboards.
VAR _CurrentRevenue = SUM(fact_sales[revenue])
VAR _PriorYearRevenue =
    CALCULATE(
        SUM(fact_sales[revenue]),
        SAMEPERIODLASTYEAR(dim_date[date])  -- Shift date context back 1 year
    )
VAR _Growth =
    DIVIDE(
        _CurrentRevenue - _PriorYearRevenue,
        _PriorYearRevenue,
        BLANK()  -- Return BLANK if prior year is 0 (avoids divide-by-zero)
    )
RETURN
    _Growth
```

**Why `DIVIDE` instead of `/`?**
- `DIVIDE(a, b, alt)` handles division by zero gracefully
- Returns `BLANK()` instead of an error
- This is a DAX best practice — never use raw division

---

## 3. Running Total (TOTALYTD)

```dax
Revenue YTD =
// Year-to-date running total of revenue.
// TOTALYTD automatically accumulates from Jan 1 to the current date.
//
// Use case: "How much revenue have we earned so far this year?"
// This resets at the start of each year automatically.
VAR _YTD =
    TOTALYTD(
        SUM(fact_sales[revenue]),
        dim_date[date]
    )
RETURN
    _YTD
```

```dax
Revenue MTD =
// Month-to-date running total.
// TOTALMTD works like TOTALYTD but resets each month.
VAR _MTD =
    TOTALMTD(
        SUM(fact_sales[revenue]),
        dim_date[date]
    )
RETURN
    _MTD
```

---

## 4. Forecast Accuracy — MAPE

```dax
MAPE =
// Mean Absolute Percentage Error — measures forecast accuracy.
//
// Formula: AVERAGE( |Actual - Predicted| / |Actual| ) × 100
//
// SUMX iterates row-by-row through fact_sales, matching each actual
// to its corresponding prediction. ABS ensures we measure magnitude
// of error, not direction.
//
// Interpretation:
//   < 10%  = Excellent forecast
//   10-20% = Good forecast
//   20-30% = Acceptable
//   > 30%  = Poor — model needs improvement
VAR _MatchedData =
    FILTER(
        fact_sales,
        // Only calculate where we have both actuals and forecasts
        CALCULATE(
            COUNTROWS(fact_forecasts),
            fact_forecasts[date_key] = fact_sales[date_key],
            fact_forecasts[store_key] = fact_sales[store_key],
            fact_forecasts[product_key] = fact_sales[product_key]
        ) > 0
    )
VAR _SumAbsError =
    SUMX(
        _MatchedData,
        VAR _Actual = fact_sales[sales]
        VAR _Predicted =
            CALCULATE(
                SUM(fact_forecasts[predicted_sales]),
                fact_forecasts[date_key] = fact_sales[date_key],
                fact_forecasts[store_key] = fact_sales[store_key],
                fact_forecasts[product_key] = fact_sales[product_key]
            )
        VAR _AbsPercentError =
            DIVIDE(
                ABS(_Actual - _Predicted),
                ABS(_Actual),
                BLANK()
            )
        RETURN
            _AbsPercentError
    )
VAR _Count = COUNTROWS(_MatchedData)
RETURN
    DIVIDE(_SumAbsError, _Count, BLANK()) * 100
```

---

## 5. % of Total (CALCULATE + ALL)

```dax
Revenue % of Total =
// Shows each row's contribution to the total.
//
// The trick: ALL(fact_sales) removes all filters from fact_sales,
// giving us the unfiltered total. CALCULATE then evaluates SUM
// in that wider context.
//
// Example: If Grocery is $5M and total is $10M → shows 50%
// This still respects slicer filters (date range, store) —
// ALL only removes the row-level filter, not external filters.
VAR _CurrentRevenue = SUM(fact_sales[revenue])
VAR _TotalRevenue =
    CALCULATE(
        SUM(fact_sales[revenue]),
        ALL(fact_sales)  -- Remove row context, keep slicer context
    )
VAR _Percentage =
    DIVIDE(
        _CurrentRevenue,
        _TotalRevenue,
        BLANK()
    )
RETURN
    _Percentage
```

```dax
Revenue % of Category =
// % contribution within the product category (not grand total).
// ALLEXCEPT keeps the category filter but removes everything else.
//
// Example: Within "Grocery", what % is "GROCERY I" vs "GROCERY II"?
VAR _CurrentRevenue = SUM(fact_sales[revenue])
VAR _CategoryRevenue =
    CALCULATE(
        SUM(fact_sales[revenue]),
        ALLEXCEPT(dim_product, dim_product[category])
    )
RETURN
    DIVIDE(_CurrentRevenue, _CategoryRevenue, BLANK())
```

---

## 6. Dynamic Measure Toggle (SWITCH + TRUE)

```dax
Selected Metric =
// Lets users switch between Revenue, Units, and Margin using a slicer.
//
// How it works:
//   1. Create a disconnected table called "Metric Selector" with 3 rows
//   2. Add it as a slicer on the report page
//   3. This measure reads the slicer selection and returns the right value
//
// SWITCH(TRUE(), ...) evaluates conditions top-to-bottom and returns
// the first match — like an if/else chain but cleaner in DAX.
//
// To create the slicer table:
//   Metric Selector = DATATABLE("Metric", STRING, {{"Revenue"}, {"Units"}, {"Margin"}})
VAR _SelectedMetric =
    SELECTEDVALUE('Metric Selector'[Metric], "Revenue")  -- Default to Revenue
VAR _Result =
    SWITCH(
        TRUE(),
        _SelectedMetric = "Revenue", SUM(fact_sales[revenue]),
        _SelectedMetric = "Units",   SUM(fact_sales[sales]),
        _SelectedMetric = "Margin",  SUM(fact_sales[margin]),
        SUM(fact_sales[revenue])  -- Fallback
    )
RETURN
    _Result
```

**Create the slicer table:**
```dax
Metric Selector = DATATABLE("Metric", STRING, {{"Revenue"}, {"Units"}, {"Margin"}})
```

---

## 7. Forecast Measures

```dax
Forecasted Revenue =
// Total forecasted revenue from the ML model predictions
VAR _ForecastRevenue = SUM(fact_forecasts[predicted_revenue])
RETURN
    _ForecastRevenue
```

```dax
Forecasted Units =
VAR _ForecastUnits = SUM(fact_forecasts[predicted_sales])
RETURN
    _ForecastUnits
```

```dax
Forecast vs Actual Variance =
// Shows the gap between what we predicted and what actually happened.
// Positive = forecast was higher than actual (over-forecast)
// Negative = forecast was lower (under-forecast)
VAR _Actual = SUM(fact_sales[revenue])
VAR _Forecast = SUM(fact_forecasts[predicted_revenue])
VAR _Variance =
    DIVIDE(
        _Forecast - _Actual,
        _Actual,
        BLANK()
    )
RETURN
    _Variance
```

---

## 8. Dynamic Forecast Metric Toggle

```dax
Selected Forecast Metric =
// Same toggle pattern as actuals, but for forecast values.
// Shares the same "Metric Selector" slicer — so switching the slicer
// updates BOTH actuals and forecast cards simultaneously.
VAR _SelectedMetric =
    SELECTEDVALUE('Metric Selector'[Metric], "Revenue")
VAR _Result =
    SWITCH(
        TRUE(),
        _SelectedMetric = "Revenue", SUM(fact_forecasts[predicted_revenue]),
        _SelectedMetric = "Units",   SUM(fact_forecasts[predicted_sales]),
        _SelectedMetric = "Margin",  SUM(fact_forecasts[predicted_margin]),
        SUM(fact_forecasts[predicted_revenue])
    )
RETURN
    _Result
```

---

## 9. Conditional Formatting Helpers

```dax
YoY Growth Color =
// Returns a color code for conditional formatting on cards/tables.
// Green for growth, red for decline, gray for flat.
VAR _Growth = [Revenue YoY Growth %]
RETURN
    SWITCH(
        TRUE(),
        _Growth > 0.05, "#059669",    -- Green: growing > 5%
        _Growth < -0.05, "#DC2626",   -- Red: declining > 5%
        "#6B7280"                       -- Gray: flat (±5%)
    )
```

```dax
MAPE Status =
// Color-code forecast accuracy for dashboard cards
VAR _MAPE = [MAPE]
RETURN
    SWITCH(
        TRUE(),
        _MAPE <= 10, "Excellent",
        _MAPE <= 20, "Good",
        _MAPE <= 30, "Acceptable",
        "Needs Improvement"
    )
```

---

## 10. Average Revenue Per Transaction

```dax
Avg Revenue Per Transaction =
// Average basket size — a key retail KPI.
// Uses DIVIDE for safe division when transactions = 0.
VAR _Revenue = SUM(fact_sales[revenue])
VAR _Transactions = SUM(fact_sales[transactions])
RETURN
    DIVIDE(_Revenue, _Transactions, BLANK())
```

---

## Measure Summary Table

| # | Measure | DAX Function | Purpose |
|---|---------|-------------|---------|
| 1 | Total Revenue | SUM | Base aggregation |
| 2 | Revenue YoY Growth % | SAMEPERIODLASTYEAR | Year-over-year comparison |
| 3 | Revenue YTD | TOTALYTD | Running year-to-date total |
| 4 | MAPE | SUMX + ABS | Forecast accuracy |
| 5 | Revenue % of Total | CALCULATE + ALL | Contribution analysis |
| 6 | Selected Metric | SWITCH(TRUE()) | Dynamic slicer toggle |
| 7 | Forecasted Revenue | SUM | Prediction aggregation |
| 8 | Forecast Variance | DIVIDE | Actual vs predicted gap |
| 9 | YoY Growth Color | SWITCH(TRUE()) | Conditional formatting |
| 10 | Avg Rev/Transaction | DIVIDE | Basket size KPI |

All measures use **VAR/RETURN** for clean, maintainable DAX.
