"""
Sales Forecasting Dashboard — Streamlit App
Interactive web dashboard for the Favorita Store Sales forecasting project.
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# Config
# ============================================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# ============================================================================
# Load Data
# ============================================================================

@st.cache_data
def load_data():
    sales = pd.read_csv(PROCESSED_DIR / "fact_sales.csv")
    forecasts = pd.read_csv(PROCESSED_DIR / "fact_forecasts.csv")
    dim_date = pd.read_csv(PROCESSED_DIR / "dim_date.csv")
    dim_store = pd.read_csv(PROCESSED_DIR / "dim_store.csv")
    dim_product = pd.read_csv(PROCESSED_DIR / "dim_product.csv")

    with open(OUTPUT_DIR / "training_results.json") as f:
        training_results = json.load(f)

    return sales, forecasts, dim_date, dim_store, dim_product, training_results


sales, forecasts, dim_date, dim_store, dim_product, training_results = load_data()

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "Dashboard",
    "Model Results",
    "Forecast Explorer",
    "Data Explorer",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Favorita Store Sales")
st.sidebar.markdown("**Dataset:** 3M+ transactions")
st.sidebar.markdown("**Best Model:** LightGBM (R² = 0.97)")
st.sidebar.markdown("**Author:** Jared Waldroff")

# ============================================================================
# Page: Dashboard
# ============================================================================

if page == "Dashboard":
    st.title("Favorita Store Sales — Forecasting Dashboard")
    st.markdown("End-to-end sales forecasting: Azure AutoML + LightGBM + Power BI")

    # --- KPI Cards ---
    total_revenue = sales["revenue"].sum()
    total_units = sales["sales"].sum()
    forecast_revenue = forecasts["predicted_revenue"].sum()
    best_r2 = training_results["metrics"]["r2"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical Revenue", f"${total_revenue / 1e9:.2f}B")
    col2.metric("Total Units Sold", f"{total_units / 1e6:.1f}M")
    col3.metric("90-Day Forecast", f"${forecast_revenue / 1e6:.0f}M")
    col4.metric("Model R²", f"{best_r2:.4f}")

    st.markdown("---")

    # --- Revenue Over Time ---
    daily_actual = sales.groupby("date_key")["revenue"].sum().reset_index()
    daily_actual["date"] = pd.to_datetime(daily_actual["date_key"], format="%Y%m%d")
    daily_actual = daily_actual.sort_values("date")

    daily_fc = forecasts.groupby("date_key")["predicted_revenue"].sum().reset_index()
    daily_fc["date"] = pd.to_datetime(daily_fc["date_key"], format="%Y%m%d")
    daily_fc = daily_fc.sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_actual["date"], y=daily_actual["revenue"],
        name="Actual Revenue", line=dict(color="#2563EB", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=daily_fc["date"], y=daily_fc["predicted_revenue"],
        name="LightGBM Forecast", line=dict(color="#DC2626", width=2, dash="dash"),
    ))
    forecast_start_ts = daily_fc["date"].min()
    fig.add_shape(
        type="line", x0=forecast_start_ts, x1=forecast_start_ts,
        y0=0, y1=1, yref="paper",
        line=dict(color="#6B7280", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=forecast_start_ts, y=1, yref="paper",
        text="Forecast starts", showarrow=False,
        font=dict(color="#6B7280", size=12),
        xanchor="left", yanchor="top",
    )
    fig.update_layout(
        title="Daily Revenue: Actuals + 90-Day Forecast",
        xaxis_title="Date", yaxis_title="Revenue ($)",
        height=450, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Two columns: Revenue by Category + Revenue by Store Type ---
    col_left, col_right = st.columns(2)

    with col_left:
        sales_with_prod = sales.merge(dim_product, on="product_key", how="left")
        cat_revenue = sales_with_prod.groupby("category")["revenue"].sum().sort_values(ascending=True).reset_index()
        fig2 = px.bar(cat_revenue, x="revenue", y="category", orientation="h",
                       title="Revenue by Product Category", color_discrete_sequence=["#2563EB"])
        fig2.update_layout(height=400, template="plotly_white", xaxis_title="Revenue ($)", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        sales_with_store = sales.merge(dim_store, on="store_key", how="left")
        type_revenue = sales_with_store.groupby("store_type")["revenue"].sum().sort_values(ascending=True).reset_index()
        fig3 = px.bar(type_revenue, x="revenue", y="store_type", orientation="h",
                       title="Revenue by Store Type", color_discrete_sequence=["#059669"])
        fig3.update_layout(height=400, template="plotly_white", xaxis_title="Revenue ($)", yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# Page: Model Results
# ============================================================================

elif page == "Model Results":
    st.title("Model Training Results")

    st.markdown("### Training Approach")
    st.markdown("""
    Models were trained two ways:
    - **Azure AutoML** (cloud): 20 models evaluated with time-series cross-validation — **Prophet** won (NRMSE = 0.0633)
    - **Local GPU Training**: 4 models on 3M rows with 90-day holdout — **LightGBM** won (NRMSE = 0.0092, R² = 0.97)
    """)

    # --- Leaderboard ---
    st.markdown("### Local Training Leaderboard")
    all_models = training_results["all_models"]
    lb_df = pd.DataFrame(all_models)
    display_cols = ["model", "normalized_rmse", "rmse", "mae", "r2", "training_time"]
    available = [c for c in display_cols if c in lb_df.columns]
    lb_display = lb_df[available].copy()
    lb_display.columns = ["Model", "Normalized RMSE", "RMSE", "MAE", "R²", "Training Time (s)"]
    lb_display = lb_display.sort_values("Normalized RMSE")
    st.dataframe(lb_display, use_container_width=True, hide_index=True)

    # --- Leaderboard Chart ---
    fig = px.bar(
        lb_display, x="Normalized RMSE", y="Model", orientation="h",
        title="Model Comparison (Normalized RMSE — lower is better)",
        color="Model",
        color_discrete_map={
            "LightGBM": "#059669",
            "XGBoost": "#2563EB",
            "RandomForest": "#2563EB",
            "ElasticNet": "#2563EB",
        },
    )
    fig.update_layout(height=350, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Feature Importance ---
    st.markdown("### Feature Importance (LightGBM)")
    try:
        import joblib
        from sklearn.preprocessing import LabelEncoder

        model = joblib.load(PROJECT_ROOT / "models" / "lightgbm_model.joblib")
        feature_cols = [
            "store_nbr", "family", "onpromotion", "city", "state", "type",
            "cluster", "oil_price", "is_holiday", "transactions",
            "year", "month", "day_of_week", "day_of_month",
            "week_of_year", "quarter", "is_weekend", "is_payday",
            "is_month_start", "oil_price_lag7",
        ]
        importances = model.feature_importances_
        imp_pct = importances / importances.sum()
        fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": imp_pct})
        fi_df = fi_df.sort_values("Importance", ascending=True)

        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                      title="Feature Importance", color_discrete_sequence=["#2563EB"])
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.image(str(PROJECT_ROOT / "screenshots" / "feature_importance.png"),
                 caption="Feature Importance (LightGBM)")

    # --- Azure AutoML Results ---
    st.markdown("### Azure AutoML Results")
    azure_data = {
        "Rank": [1, 2, 3, 4, 5],
        "Model": ["Prophet", "Exponential Smoothing", "Voting Ensemble", "XGBoost", "LightGBM"],
        "Normalized RMSE": [0.0633, 0.0641, 0.0645, 0.0658, 0.0672],
    }
    st.dataframe(pd.DataFrame(azure_data), use_container_width=True, hide_index=True)
    st.caption("Azure AutoML used 3-fold rolling-origin cross-validation, producing higher NRMSE than single holdout splits.")

# ============================================================================
# Page: Forecast Explorer
# ============================================================================

elif page == "Forecast Explorer":
    st.title("90-Day Forecast Explorer")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.selectbox("Forecast Horizon", ["All", "30-Day", "60-Day", "90-Day"])
    with col2:
        cities = ["All"] + sorted(dim_store["city"].unique().tolist())
        selected_city = st.selectbox("City", cities)

    # Filter forecasts
    fc = forecasts.copy()
    if horizon != "All":
        fc = fc[fc["forecast_horizon"] == horizon]
    if selected_city != "All":
        city_stores = dim_store[dim_store["city"] == selected_city]["store_key"].tolist()
        fc = fc[fc["store_key"].isin(city_stores)]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast Revenue", f"${fc['predicted_revenue'].sum() / 1e6:.1f}M")
    col2.metric("Forecast Units", f"{fc['predicted_sales'].sum() / 1e3:.0f}K")
    col3.metric("Forecast Margin", f"${fc['predicted_margin'].sum() / 1e6:.1f}M")

    # Daily forecast chart
    daily_fc = fc.groupby("date_key").agg(
        revenue=("predicted_revenue", "sum"),
        units=("predicted_sales", "sum"),
    ).reset_index()
    daily_fc["date"] = pd.to_datetime(daily_fc["date_key"], format="%Y%m%d")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_fc["date"], y=daily_fc["revenue"],
        name="Predicted Revenue", fill="tozeroy",
        line=dict(color="#DC2626"),
    ))
    fig.update_layout(
        title="Daily Forecasted Revenue",
        xaxis_title="Date", yaxis_title="Revenue ($)",
        height=400, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast by product category
    fc_with_prod = fc.merge(dim_product, on="product_key", how="left")
    cat_fc = fc_with_prod.groupby("category")["predicted_revenue"].sum().sort_values(ascending=True).reset_index()
    fig2 = px.bar(cat_fc, x="predicted_revenue", y="category", orientation="h",
                   title="Forecasted Revenue by Category", color_discrete_sequence=["#DC2626"])
    fig2.update_layout(height=400, template="plotly_white", xaxis_title="Predicted Revenue ($)", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

    # Forecast by city
    fc_with_store = fc.merge(dim_store, on="store_key", how="left")
    city_fc = fc_with_store.groupby("city")["predicted_revenue"].sum().sort_values(ascending=False).head(10).reset_index()
    fig3 = px.bar(city_fc, x="predicted_revenue", y="city", orientation="h",
                   title="Top 10 Cities by Forecasted Revenue", color_discrete_sequence=["#059669"])
    fig3.update_layout(height=400, template="plotly_white", xaxis_title="Predicted Revenue ($)", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# Page: Data Explorer
# ============================================================================

elif page == "Data Explorer":
    st.title("Data Explorer")

    tab1, tab2, tab3, tab4 = st.tabs(["Sales", "Forecasts", "Stores", "Products"])

    with tab1:
        st.markdown(f"**fact_sales**: {len(sales):,} rows")
        year_filter = st.selectbox("Filter by year", ["All"] + sorted(dim_date["year"].unique().tolist()))
        display_sales = sales.head(1000)
        if year_filter != "All":
            year_keys = dim_date[dim_date["year"] == year_filter]["date_key"].tolist()
            display_sales = sales[sales["date_key"].isin(year_keys)].head(1000)
        st.dataframe(display_sales, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown(f"**fact_forecasts**: {len(forecasts):,} rows")
        horizon_filter = st.selectbox("Filter by horizon", ["All", "30-Day", "60-Day", "90-Day"], key="fc_horizon")
        display_fc = forecasts if horizon_filter == "All" else forecasts[forecasts["forecast_horizon"] == horizon_filter]
        st.dataframe(display_fc.head(1000), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown(f"**dim_store**: {len(dim_store)} stores")
        st.dataframe(dim_store, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown(f"**dim_product**: {len(dim_product)} product families")
        st.dataframe(dim_product, use_container_width=True, hide_index=True)
