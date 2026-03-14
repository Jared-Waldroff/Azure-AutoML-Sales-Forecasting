"""
============================================================================
Model Deployment Pipeline
============================================================================
Deploys the best AutoML forecasting model as a managed online endpoint.

Why managed online endpoints?
    - Azure handles scaling, load balancing, and authentication
    - HTTPS endpoint with built-in auth tokens
    - Blue/green deployment support for zero-downtime updates
    - Auto-scaling based on request volume
    - Pay only for compute when the endpoint is active

Pipeline Steps:
    1. Connect to workspace and retrieve best model
    2. Register the model as a versioned asset
    3. Create a managed online endpoint
    4. Deploy the model to the endpoint
    5. Test with a sample request

Usage:
    python src/deploy_endpoint.py --job-name <automl-job-name>

Author: Jared Waldroff
============================================================================
"""

import os
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
)

# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "sales-forecast-endpoint")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "sales-forecast-deploy")
DEPLOYMENT_INSTANCE_TYPE = os.getenv("DEPLOYMENT_INSTANCE_TYPE", "Standard_DS3_v2")
DEPLOYMENT_INSTANCE_COUNT = int(os.getenv("DEPLOYMENT_INSTANCE_COUNT", "1"))


def get_ml_client() -> MLClient:
    """Connect to Azure ML workspace."""
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
        workspace_name=os.getenv("AML_WORKSPACE_NAME"),
    )


# ============================================================================
# Step 1: Retrieve Best Model
# ============================================================================

def get_best_model_from_automl(ml_client: MLClient, job_name: str) -> str:
    """
    Find the best model from an AutoML experiment and register it.

    Azure AutoML automatically registers the best model, but we
    re-register it with a descriptive name and tags for clarity.
    Tags are searchable in Azure ML Studio — useful when you have
    many models across experiments.

    Args:
        ml_client: Authenticated Azure ML client
        job_name: Name of the completed AutoML experiment

    Returns:
        str: Registered model name with version
    """
    print("=" * 60)
    print("STEP 1: Retrieving Best Model")
    print("=" * 60)

    # Get the AutoML parent job
    automl_job = ml_client.jobs.get(job_name)

    # The best model's run ID is stored in the parent job's properties
    best_child_run_id = automl_job.properties.get("best_child_run_id")

    if not best_child_run_id:
        # Fallback: iterate child runs to find the best
        best_score = float("inf")
        for child in ml_client.jobs.list(parent_job_name=job_name):
            if child.status == "Completed":
                score = float(child.properties.get("score", float("inf")))
                if score < best_score:
                    best_score = score
                    best_child_run_id = child.name

    print(f"  Best model run: {best_child_run_id}")

    # Register the model with descriptive metadata
    model = Model(
        name="store-sales-forecaster",
        description="AutoML time-series forecasting model for Favorita store sales",
        path=f"azureml://jobs/{best_child_run_id}/outputs/artifacts/outputs/model/",
        type="mlflow_model",  # AutoML outputs MLflow models
        tags={
            "project": "sales-forecasting",
            "dataset": "favorita-store-sales",
            "horizon": "90-days",
            "trained_by": "azure-automl",
        },
    )

    registered_model = ml_client.models.create_or_update(model)
    model_id = f"{registered_model.name}:{registered_model.version}"

    print(f"  ✓ Model registered: {model_id}")

    return model_id


# ============================================================================
# Step 2: Create Managed Online Endpoint
# ============================================================================

def create_endpoint(ml_client: MLClient) -> str:
    """
    Create a managed online endpoint for real-time predictions.

    Endpoint vs Deployment:
        - Endpoint: The URL (like a domain name). Receives traffic.
        - Deployment: The actual model instance behind the endpoint.
        - One endpoint can have multiple deployments (blue/green pattern).

    Authentication:
        - auth_mode="key": Simple API key authentication
        - Alternative: "aml_token" for Azure AD-based auth
        - Keys are auto-generated and rotatable via Azure ML Studio

    Args:
        ml_client: Authenticated Azure ML client

    Returns:
        str: Endpoint name
    """
    print("\n" + "=" * 60)
    print("STEP 2: Creating Online Endpoint")
    print("=" * 60)

    try:
        # Check if endpoint already exists
        existing = ml_client.online_endpoints.get(ENDPOINT_NAME)
        print(f"  ✓ Endpoint already exists: {ENDPOINT_NAME}")
        print(f"    Scoring URI: {existing.scoring_uri}")
        return ENDPOINT_NAME

    except Exception:
        # Create new endpoint
        print(f"  Creating endpoint: {ENDPOINT_NAME}...")

        endpoint = ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            description="Real-time sales forecasting endpoint",
            auth_mode="key",  # API key authentication
            tags={
                "project": "sales-forecasting",
                "environment": "production",
            },
        )

        # begin_create_or_update returns a poller
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        print(f"  ✓ Endpoint created: {ENDPOINT_NAME}")

    return ENDPOINT_NAME


# ============================================================================
# Step 3: Deploy Model to Endpoint
# ============================================================================

def deploy_model(ml_client: MLClient, model_id: str, endpoint_name: str):
    """
    Deploy the registered model to the online endpoint.

    Deployment Configuration:
        - instance_type: VM size for serving. DS3_v2 handles ~100 req/sec.
        - instance_count: Number of replicas. 1 is fine for demo.
        - In production, you'd set auto-scaling rules based on traffic.

    MLflow Deployment:
        AutoML models are saved in MLflow format, which Azure ML can
        serve natively without a custom scoring script. This means:
        - No score.py needed (MLflow handles serialization)
        - No environment.yml needed (dependencies are in the model)
        - Simpler deployment, fewer things to break

    Args:
        ml_client: Authenticated Azure ML client
        model_id: Registered model name:version
        endpoint_name: Target endpoint name
    """
    print("\n" + "=" * 60)
    print("STEP 3: Deploying Model")
    print("=" * 60)

    model_name, model_version = model_id.split(":")

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=endpoint_name,
        model=f"azureml:{model_name}:{model_version}",
        instance_type=DEPLOYMENT_INSTANCE_TYPE,
        instance_count=DEPLOYMENT_INSTANCE_COUNT,
    )

    print(f"  Deploying {model_id} to {endpoint_name}...")
    print(f"    Instance type: {DEPLOYMENT_INSTANCE_TYPE}")
    print(f"    Instance count: {DEPLOYMENT_INSTANCE_COUNT}")
    print(f"    (This may take 5-10 minutes...)")

    # Deploy — this provisions the VM and loads the model
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Route 100% of traffic to this deployment
    # In a blue/green scenario, you'd split traffic gradually
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"  ✓ Deployment complete!")
    print(f"    Endpoint: {endpoint_name}")
    print(f"    Traffic: 100% → {DEPLOYMENT_NAME}")


# ============================================================================
# Step 4: Test the Endpoint
# ============================================================================

def test_endpoint(ml_client: MLClient, endpoint_name: str):
    """
    Send a test prediction request to verify the deployment works.

    The input format follows the MLflow serving convention:
        - "input_data": Contains the feature DataFrame as records
        - Each record represents one time step to forecast
        - Required columns match the training data schema

    This test request predicts sales for 3 stores × 1 product family
    over the next few days. In production, you'd batch-score all
    store×product combinations.

    Args:
        ml_client: Authenticated Azure ML client
        endpoint_name: Deployed endpoint name
    """
    print("\n" + "=" * 60)
    print("STEP 4: Testing Endpoint")
    print("=" * 60)

    # Construct a sample scoring request
    # This mimics what score_forecasts.py will send at scale
    sample_request = {
        "input_data": {
            "columns": [
                "date", "store_nbr", "family", "onpromotion",
                "oil_price", "is_holiday", "transactions",
                "year", "month", "day_of_week", "day_of_month",
                "week_of_year", "quarter", "is_weekend", "is_payday",
                "is_month_start", "oil_price_lag7",
                "city", "state", "type", "cluster"
            ],
            "data": [
                # Store 1, GROCERY I, a typical Monday
                ["2017-08-16", 1, "GROCERY I", 50,
                 55.0, 0, 2500,
                 2017, 8, 2, 16,
                 33, 3, 0, 1,
                 0, 54.5,
                 "Quito", "Pichincha", "D", 13],
                # Store 1, GROCERY I, next day (Tuesday)
                ["2017-08-17", 1, "GROCERY I", 48,
                 55.2, 0, 2450,
                 2017, 8, 3, 17,
                 33, 3, 0, 0,
                 0, 54.8,
                 "Quito", "Pichincha", "D", 13],
            ]
        }
    }

    # Save request to file (required by invoke method)
    request_path = PROJECT_ROOT / "data" / "output" / "test_request.json"
    with open(request_path, "w") as f:
        json.dump(sample_request, f, indent=2)

    print(f"  Sending test request to {endpoint_name}...")

    try:
        # Invoke the endpoint
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=str(request_path),
        )

        predictions = json.loads(response)

        print(f"  ✓ Endpoint responded successfully!")
        print(f"    Predictions: {predictions}")

        # Save test response
        response_path = PROJECT_ROOT / "data" / "output" / "test_response.json"
        with open(response_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"    Saved to: {response_path}")

    except Exception as e:
        print(f"  ⚠ Test request failed: {e}")
        print(f"    This is expected if the endpoint is still warming up.")
        print(f"    Try again in a few minutes, or test via Azure ML Studio.")


# ============================================================================
# Main
# ============================================================================

def main():
    """Deploy the best AutoML model as a real-time endpoint."""
    parser = argparse.ArgumentParser(description="Deploy forecasting model to endpoint")
    parser.add_argument("--job-name", type=str, required=True,
                        help="AutoML job name from training step")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Sales Forecasting - Model Deployment                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    ml_client = get_ml_client()

    # Pipeline: Retrieve → Endpoint → Deploy → Test
    model_id = get_best_model_from_automl(ml_client, args.job_name)
    endpoint_name = create_endpoint(ml_client)
    deploy_model(ml_client, model_id, endpoint_name)
    test_endpoint(ml_client, endpoint_name)

    # Print the scoring URI for use in score_forecasts.py
    endpoint = ml_client.online_endpoints.get(endpoint_name)

    print("\n" + "=" * 60)
    print("✓ DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"  Endpoint: {endpoint_name}")
    print(f"  Scoring URI: {endpoint.scoring_uri}")
    print(f"\n  Next step: Run src/score_forecasts.py to generate predictions")
    print(f"  Then: Import forecast data into Power BI")


if __name__ == "__main__":
    main()
