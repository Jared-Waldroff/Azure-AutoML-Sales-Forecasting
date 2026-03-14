# Power Query (M) Transformations

## Overview

Power Query handles the ETL (Extract, Transform, Load) layer in Power BI.
These M code transformations shape the raw CSV imports into the optimized
format our data model needs.

Apply these in **Power BI Desktop → Transform Data → Advanced Editor**.

---

## 1. fact_sales Transformation

```m
let
    // Step 1: Load the CSV file
    // Change the file path to match your local setup
    Source = Csv.Document(
        File.Contents("C:\path\to\data\processed\fact_sales.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),

    // Step 2: Promote first row to column headers
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),

    // Step 3: Set column data types
    // Explicit typing prevents auto-detection errors and improves compression
    TypedColumns = Table.TransformColumnTypes(PromotedHeaders, {
        {"date_key", Int64.Type},
        {"store_key", Int64.Type},
        {"product_key", Int64.Type},
        {"sales", type number},
        {"revenue", Currency.Type},          // Currency type formats as $
        {"margin", Currency.Type},
        {"onpromotion", Int64.Type},
        {"oil_price", type number},
        {"is_holiday", Int64.Type},
        {"transactions", Int64.Type}
    }),

    // Step 4: Remove the raw date column (we use date_key → dim_date instead)
    // This avoids duplicate date columns and keeps the star schema clean
    RemovedDate = Table.RemoveColumns(TypedColumns, {"date"}, MissingField.Ignore),

    // Step 5: Add a calculated column for "Has Promotion" flag
    // Boolean columns compress better and are faster to filter
    AddedPromoFlag = Table.AddColumn(
        RemovedDate,
        "has_promotion",
        each if [onpromotion] > 0 then true else false,
        type logical
    )
in
    AddedPromoFlag
```

---

## 2. fact_forecasts Transformation

```m
let
    Source = Csv.Document(
        File.Contents("C:\path\to\data\processed\fact_forecasts.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),

    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),

    // Type the columns to match fact_sales schema
    TypedColumns = Table.TransformColumnTypes(PromotedHeaders, {
        {"date_key", Int64.Type},
        {"store_key", Int64.Type},
        {"product_key", Int64.Type},
        {"predicted_sales", type number},
        {"predicted_revenue", Currency.Type},
        {"predicted_margin", Currency.Type},
        {"forecast_horizon", type text}
    }),

    // Remove raw date column (use date_key instead)
    RemovedDate = Table.RemoveColumns(TypedColumns, {"date"}, MissingField.Ignore),

    // Add a "data_source" column to distinguish from actuals in combined views
    AddedSource = Table.AddColumn(
        RemovedDate,
        "data_source",
        each "Forecast",
        type text
    )
in
    AddedSource
```

---

## 3. dim_date Transformation

```m
let
    Source = Csv.Document(
        File.Contents("C:\path\to\data\processed\dim_date.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),

    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),

    TypedColumns = Table.TransformColumnTypes(PromotedHeaders, {
        {"date_key", Int64.Type},
        {"date", type date},
        {"year", Int64.Type},
        {"quarter", Int64.Type},
        {"month", Int64.Type},
        {"month_name", type text},
        {"week_of_year", Int64.Type},
        {"day_of_week", Int64.Type},
        {"day_name", type text},
        {"day_of_month", Int64.Type},
        {"is_weekend", Int64.Type},
        {"fiscal_year", Int64.Type},
        {"year_month", type text},
        {"year_quarter", type text},
        {"is_holiday", Int64.Type}
    }),

    // Add a sort column for month_name (so Jan sorts before Feb, not alphabetically)
    // Without this, Power BI would sort month_name alphabetically:
    // April, August, December... instead of January, February, March...
    AddedMonthSort = Table.AddColumn(
        TypedColumns,
        "month_sort",
        each [month],
        Int64.Type
    ),

    // Add a sort column for day_name
    AddedDaySort = Table.AddColumn(
        AddedMonthSort,
        "day_sort",
        each [day_of_week],
        Int64.Type
    )
in
    AddedDaySort
```

**After loading:** Set the sort-by-column property:
- Select `month_name` → Column tools → Sort by column → `month_sort`
- Select `day_name` → Column tools → Sort by column → `day_sort`

---

## 4. dim_store Transformation

```m
let
    Source = Csv.Document(
        File.Contents("C:\path\to\data\processed\dim_store.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),

    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),

    TypedColumns = Table.TransformColumnTypes(PromotedHeaders, {
        {"store_key", Int64.Type},
        {"store_nbr", Int64.Type},
        {"city", type text},
        {"state", type text},
        {"store_type", type text},
        {"cluster", Int64.Type},
        {"region", type text}
    }),

    // Add a display name combining store number and city for better labels
    AddedDisplayName = Table.AddColumn(
        TypedColumns,
        "store_label",
        each "Store " & Text.From([store_nbr]) & " - " & [city],
        type text
    )
in
    AddedDisplayName
```

---

## 5. dim_product Transformation

```m
let
    Source = Csv.Document(
        File.Contents("C:\path\to\data\processed\dim_product.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),

    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),

    TypedColumns = Table.TransformColumnTypes(PromotedHeaders, {
        {"product_key", Int64.Type},
        {"family", type text},
        {"category", type text}
    })
in
    TypedColumns
```

---

## 6. Metric Selector Table (Disconnected)

Create this as a **New Table** in Power BI (not imported):

```dax
Metric Selector = DATATABLE("Metric", STRING, {{"Revenue"}, {"Units"}, {"Margin"}})
```

This table is "disconnected" — it has no relationships to any other table.
It drives the `Selected Metric` and `Selected Forecast Metric` DAX measures
via `SELECTEDVALUE()`.

---

## Tips for Power Query Performance

1. **Type columns early** — Vertipaq compresses typed columns much better
2. **Remove unused columns** — Every column costs memory and refresh time
3. **Use Int64 for keys** — Integer keys compress and join faster than strings
4. **Avoid calculated columns in PQ when DAX works** — PQ columns increase model size; DAX measures compute on the fly
5. **Disable auto date/time** — File → Options → Current File → Data Load → uncheck "Auto date/time for new files" (our dim_date handles this)
