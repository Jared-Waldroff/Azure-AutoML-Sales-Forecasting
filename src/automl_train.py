"""
============================================================================
Azure AutoML Training Pipeline
============================================================================
Configures and submits a time-series forecasting experiment to Azure AutoML.

This script handles the full ML lifecycle:
    1. Connect to Azure ML workspace (authenticates via DefaultAzureCredential)
    2. Create or reuse a compute cluster (auto-scaling, cost-optimized)
    3. Upload training data as an Azure ML dataset
    4. Configure AutoML forecasting (horizon, metrics, allowed models)
    5. Submit the experiment and monitor progress
    6. Retrieve the best model and its performance metrics

Key Design Decisions:
    - We use Azure ML SDK v2 (azure-ai-ml) — the modern, recommended SDK
    - Compute auto-scales to 0 when idle to minimize costs
    - Training timeout prevents runaway costs
    - All config is externalized to .env for portability

Usage:
    python src/automl_train.py

Prerequisites:
    - Azure ML workspace provisioned (see docs/architecture.md)
    - Data prepared (run src/data_prep.py first)
    - Azure CLI authenticated: az login

Author: Jared Waldroff
============================================================================
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, automl
from azure.ai.ml.entities import (
    AmlCompute,
    Data,
)
from azure.ai.ml.constants import AssetTypes

# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Azure configuration from .env
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("AML_WORKSPACE_NAME")
COMPUTE_NAME = os.getenv("AML_COMPUTE_NAME", "forecast-cluster")
COMPUTE_SIZE = os.getenv("AML_COMPUTE_SIZE", "Standard_DS3_v2")
MIN_NODES = int(os.getenv("AML_COMPUTE_MIN_NODES", "0"))
MAX_NODES = int(os.getenv("AML_COMPUTE_MAX_NODES", "4"))
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "store-sales-forecasting")
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "90"))
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "normalized_root_mean_squared_error")
TIMEOUT_HOURS = int(os.getenv("TRAINING_TIMEOUT_HOURS", "2"))


# ============================================================================
# Step 1: Connect to Azure ML Workspace
# ============================================================================

def get_ml_client() -> MLClient:
    """
    Authenticate and create an Azure ML client connection.

    Uses DefaultAzureCredential which tries multiple auth methods in order:
        1. Environment variables (AZURE_CLIENT_ID, etc.)
        2. Managed identity (if running in Azure)
        3. Azure CLI (az login) — most common for development
        4. Visual Studio Code credentials
        5. Interactive browser login (fallback)

    This chain means the same code works locally, in CI/CD, and in Azure
    without code changes — just different credential configurations.

    Returns:
        MLClient: Authenticated client for Azure ML operations
    """
    print("=" * 60)
    print("STEP 1: Connecting to Azure ML Workspace")
    print("=" * 60)

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    # Verify connection by fetching workspace details
    ws = ml_client.workspaces.get(WORKSPACE_NAME)
    print(f"  ✓ Connected to workspace: {ws.name}")
    print(f"    Resource group: {RESOURCE_GROUP}")
    print(f"    Region: {ws.location}")

    return ml_client


# ============================================================================
# Step 2: Create or Get Compute Cluster
# ============================================================================

def setup_compute(ml_client: MLClient) -> str:
    """
    Create an auto-scaling compute cluster for AutoML training.

    Why a cluster (not a compute instance)?
        - AutoML runs multiple child models in parallel
        - Cluster auto-scales: 0 nodes when idle = $0 cost
        - Multiple nodes accelerate the model search significantly

    Cost optimization:
        - min_nodes=0: Cluster scales to zero when not training
        - Standard_DS3_v2: ~$0.24/hr per node — good balance of cost/performance
        - With 4 max nodes and 2hr timeout: worst case ~$1.92 total

    Args:
        ml_client: Authenticated Azure ML client

    Returns:
        str: Name of the compute target
    """
    print("\n" + "=" * 60)
    print("STEP 2: Setting Up Compute Cluster")
    print("=" * 60)

    try:
        # Check if compute already exists (reuse to save provisioning time)
        compute = ml_client.compute.get(COMPUTE_NAME)
        print(f"  ✓ Found existing compute: {COMPUTE_NAME}")
        print(f"    Size: {compute.size} | Nodes: {compute.min_instances}-{compute.max_instances}")

    except Exception:
        # Compute doesn't exist — create it
        print(f"  Creating compute cluster: {COMPUTE_NAME}...")

        compute = AmlCompute(
            name=COMPUTE_NAME,
            size=COMPUTE_SIZE,
            min_instances=MIN_NODES,        # Scale to 0 when idle
            max_instances=MAX_NODES,         # Scale up for parallel training
            idle_time_before_scale_down=300, # 5 min idle → scale down
        )

        # begin_create_or_update returns a poller; .result() waits for completion
        ml_client.compute.begin_create_or_update(compute).result()

        print(f"  ✓ Compute cluster created: {COMPUTE_NAME}")
        print(f"    VM Size: {COMPUTE_SIZE}")
        print(f"    Auto-scale: {MIN_NODES} → {MAX_NODES} nodes")
        print(f"    Idle timeout: 300s (scales to 0 to save costs)")

    return COMPUTE_NAME


# ============================================================================
# Step 3: Register Training Data
# ============================================================================

def register_training_data(ml_client: MLClient) -> Input:
    """
    Upload training data to Azure ML as a registered dataset.

    Registering data as an Azure ML asset provides:
        - Versioning: Track which data trained which model
        - Lineage: Audit trail from raw data → model → predictions
        - Reuse: Other experiments can reference the same data

    We upload the Parquet file because:
        - Parquet preserves column types (no CSV parsing issues)
        - 5-10x smaller than CSV (faster upload/download)
        - Columnar format = faster reads for specific columns

    Args:
        ml_client: Authenticated Azure ML client

    Returns:
        Input: Reference to the registered dataset for AutoML
    """
    print("\n" + "=" * 60)
    print("STEP 3: Registering Training Data")
    print("=" * 60)

    # AutoML requires MLTable format — a folder with an MLTable YAML
    # definition file that points to the actual Parquet data.
    # MLTable tells Azure how to read and interpret the data schema.
    mltable_path = PROCESSED_DIR / "training_mltable"

    if not (mltable_path / "MLTable").exists():
        raise FileNotFoundError(
            f"MLTable not found at {mltable_path}. "
            "Run src/data_prep.py first."
        )

    # Register as an MLTable data asset (required by AutoML)
    data_asset = Data(
        name="store-sales-training-mltable",
        description="Favorita store sales with engineered features for forecasting (MLTable)",
        path=str(mltable_path),
        type=AssetTypes.MLTABLE,
    )

    registered_data = ml_client.data.create_or_update(data_asset)

    print(f"  ✓ Data registered: {registered_data.name} v{registered_data.version}")
    print(f"    Source: {mltable_path.name}")

    # Return an Input reference for AutoML to consume
    return Input(
        type=AssetTypes.MLTABLE,
        path=f"azureml:{registered_data.name}:{registered_data.version}"
    )


# ============================================================================
# Step 4: Configure AutoML Forecasting
# ============================================================================

def configure_automl(training_data: Input, compute_name: str) -> automl.ForecastingJob:
    """
    Configure the AutoML forecasting experiment.

    AutoML Forecasting Key Concepts:
        - time_column_name: The datetime column AutoML uses for ordering
        - time_series_id: Columns that identify unique time series
          (here: store_nbr + family = one series per store×product combo)
        - forecast_horizon: How far ahead to predict (90 days)
        - target_lags: AutoML auto-generates lag features from the target
        - target_rolling_window_size: Rolling aggregations of past values

    Model Search:
        AutoML will try these model families automatically:
        - Statistical: ARIMA, ETS, Seasonal Naive, Theta
        - Tree-based: LightGBM, XGBoost (often winners for retail data)
        - Deep Learning: TCNForecaster (temporal conv net)
        - Prophet: Facebook's additive model (good with holidays)

    Why normalized_root_mean_squared_error?
        - Normalized RMSE allows comparison across series with different scales
        - A store selling 10,000 units vs 100 units need comparable metrics
        - Alternative: normalized_mean_absolute_error (more robust to outliers)

    Args:
        training_data: Registered dataset reference
        compute_name: Name of the compute target

    Returns:
        ForecastingJob: Configured AutoML job ready for submission
    """
    print("\n" + "=" * 60)
    print("STEP 4: Configuring AutoML Forecasting")
    print("=" * 60)

    # Create the forecasting job configuration
    # When using serverless compute, we omit the compute parameter entirely
    # Azure will automatically provision and manage the VMs
    use_serverless = (compute_name == "serverless")

    # Build keyword args — only include compute if we have a dedicated cluster
    job_kwargs = {
        "training_data": training_data,
        "target_column_name": "sales",
        "primary_metric": PRIMARY_METRIC,
        "enable_model_explainability": True,
        # Time-series forecasting requires either cross-validation or a validation set
        # 3-fold rolling-origin CV is standard for time-series (respects temporal order)
        "n_cross_validations": 3,
    }
    if not use_serverless:
        job_kwargs["compute"] = compute_name

    # Configure forecasting settings upfront
    # ForecastingSettings holds all time-series-specific configuration
    forecast_settings = automl.ForecastingSettings(
        time_column_name="date",
        time_series_id_column_names=["store_nbr", "family"],  # Each store×product = unique series
        forecast_horizon=FORECAST_HORIZON,                      # 90 days ahead
        frequency="D",                                          # Daily granularity
        target_lags="auto",                                     # AutoML determines optimal lags
        target_rolling_window_size="auto",                      # AutoML determines window sizes
    )
    job_kwargs["forecasting_settings"] = forecast_settings

    forecasting_job = automl.forecasting(**job_kwargs)

    # --- Experiment name ---
    forecasting_job.experiment_name = EXPERIMENT_NAME

    # --- Training Settings ---
    # Configure which algorithms to try and enable deep learning
    forecasting_job.set_training(
        enable_dnn_training=True,           # Include deep learning models (TCN)
        allowed_training_algorithms=[
            # Let AutoML try a wide range — the leaderboard shows what works
            "LightGBM", "XGBoostRegressor", "Prophet",
            "AutoArima", "ExponentialSmoothing",
            "TCNForecaster", "ElasticNet",
            "DecisionTree", "RandomForest",
        ],
    )

    # --- Resource Limits ---
    forecasting_job.set_limits(
        timeout_minutes=TIMEOUT_HOURS * 60,    # Max total training time
        trial_timeout_minutes=30,               # Max per individual model
        max_trials=50,                          # Max models to try
        max_concurrent_trials=2,                # Match our 2-node cluster
        enable_early_termination=True,          # Stop bad models early
    )

    # --- Featurization ---
    # "auto" lets AutoML handle:
    #   - Missing value imputation
    #   - Categorical encoding
    #   - Date feature extraction (it generates its own calendar features)
    #   - Feature normalization
    forecasting_job.set_featurization(mode="auto")

    compute_label = "serverless" if use_serverless else compute_name
    print(f"  ✓ AutoML configuration complete:")
    print(f"    Target: sales")
    print(f"    Metric: {PRIMARY_METRIC}")
    print(f"    Compute: {compute_label}")
    print(f"    Horizon: {FORECAST_HORIZON} days")
    print(f"    Series IDs: store_nbr × family")
    print(f"    Max trials: 50 | Timeout: {TIMEOUT_HOURS}hr")
    print(f"    DNN training: enabled")
    print(f"    Explainability: enabled")

    return forecasting_job


# ============================================================================
# Step 5: Submit and Monitor Experiment
# ============================================================================

def submit_experiment(ml_client: MLClient, job: automl.ForecastingJob) -> str:
    """
    Submit the AutoML job and monitor progress until completion.

    What happens when you submit:
        1. Azure provisions compute nodes (if cluster was at 0)
        2. AutoML creates a data profile (stats, distributions)
        3. Featurization pipeline runs (encoding, imputation)
        4. Model search begins — each "trial" trains one model variant
        5. Models are ranked by primary metric on validation set
        6. Best model is registered with its metrics and artifacts

    The experiment is viewable in Azure ML Studio at:
        https://ml.azure.com → Experiments → {EXPERIMENT_NAME}

    Args:
        ml_client: Authenticated Azure ML client
        job: Configured forecasting job

    Returns:
        str: Job name/ID for retrieving results
    """
    print("\n" + "=" * 60)
    print("STEP 5: Submitting AutoML Experiment")
    print("=" * 60)

    # Submit the job — this returns immediately with a job reference
    submitted_job = ml_client.jobs.create_or_update(job)

    print(f"  ✓ Job submitted: {submitted_job.name}")
    print(f"    Status: {submitted_job.status}")
    print(f"    Studio URL: {submitted_job.studio_url}")
    print(f"\n  ⏳ Waiting for completion (this may take 1-2 hours)...")
    print(f"     Monitor in Azure ML Studio: {submitted_job.studio_url}")

    # Poll for completion
    # In production, you'd use webhooks or Azure Event Grid
    # For this project, polling is simpler and more portable
    while True:
        current_job = ml_client.jobs.get(submitted_job.name)
        status = current_job.status

        if status in ["Completed", "Failed", "Canceled"]:
            break

        print(f"    Status: {status}...", end="\r")
        time.sleep(60)  # Check every minute

    if status == "Completed":
        print(f"\n  ✓ Experiment completed successfully!")
    else:
        print(f"\n  ✗ Experiment ended with status: {status}")
        raise RuntimeError(f"AutoML job failed with status: {status}")

    return submitted_job.name


# ============================================================================
# Step 6: Retrieve Best Model Results
# ============================================================================

def get_best_model(ml_client: MLClient, job_name: str):
    """
    Retrieve the best model from the completed AutoML experiment.

    AutoML ranks all trained models by the primary metric and selects
    the winner. This function extracts:
        - Model type and hyperparameters
        - Validation metrics (RMSE, MAE, MAPE, R²)
        - Feature importance rankings

    This information feeds directly into:
        - docs/model_selection.md (for the README)
        - The "model rationale" section of your employer presentation

    Args:
        ml_client: Authenticated Azure ML client
        job_name: Name of the completed AutoML job

    Returns:
        dict: Best model details including metrics and feature importance
    """
    print("\n" + "=" * 60)
    print("STEP 6: Retrieving Best Model")
    print("=" * 60)

    # Get the parent job (AutoML experiment)
    automl_job = ml_client.jobs.get(job_name)

    # Get the best child run (the winning model)
    best_run = ml_client.jobs.list(
        parent_job_name=job_name,
    )

    # The first result when sorting by primary metric is the best
    best_child = None
    best_metric = float('inf')

    for child in best_run:
        if child.status == "Completed" and hasattr(child, 'properties'):
            metric_val = child.properties.get('score', float('inf'))
            try:
                if float(metric_val) < best_metric:
                    best_metric = float(metric_val)
                    best_child = child
            except (ValueError, TypeError):
                continue

    if best_child:
        print(f"  ✓ Best model found:")
        print(f"    Run ID: {best_child.name}")
        print(f"    Score ({PRIMARY_METRIC}): {best_metric:.6f}")
        print(f"    Properties: {best_child.properties}")

    # Save results for documentation
    results = {
        "job_name": job_name,
        "best_run_id": best_child.name if best_child else "N/A",
        "primary_metric": PRIMARY_METRIC,
        "best_score": best_metric,
        "studio_url": automl_job.studio_url,
    }

    print(f"\n  📊 View full leaderboard in Azure ML Studio:")
    print(f"     {automl_job.studio_url}")
    print(f"\n  Next step: Run src/model_evaluate.py for detailed analysis")
    print(f"  Then: Run src/deploy_endpoint.py to deploy the winner")

    return results


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """
    Orchestrate the full AutoML training pipeline.

    End-to-end flow:
        Connect → Compute → Data → Configure → Submit → Results
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Sales Forecasting - Azure AutoML Training Pipeline   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Validate config
    if not SUBSCRIPTION_ID or SUBSCRIPTION_ID == "your-subscription-id-here":
        print("\n  ✗ ERROR: Azure credentials not configured.")
        print("    1. Copy .env.template to .env")
        print("    2. Fill in your Azure subscription details")
        print("    3. Run: az login")
        return

    # Execute pipeline
    ml_client = get_ml_client()

    # Use the existing compute cluster (Standard_D2as_v4, DASv4 family)
    # This family has quota available on our subscription
    compute_name = "forecast-cluster"
    print(f"\n  Using compute cluster: {compute_name}")

    training_data = register_training_data(ml_client)
    forecasting_job = configure_automl(training_data, compute_name)
    job_name = submit_experiment(ml_client, forecasting_job)
    results = get_best_model(ml_client, job_name)

    print("\n" + "=" * 60)
    print("✓ AUTOML TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best score: {results['best_score']:.6f} ({PRIMARY_METRIC})")
    print(f"  Studio URL: {results['studio_url']}")


if __name__ == "__main__":
    main()
