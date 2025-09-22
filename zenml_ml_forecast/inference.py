import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.kubernetes.flavors import KubernetesOrchestratorSettings
from zenml.integrations.kubernetes.pod_settings import KubernetesPodSettings

from zenml_ml_forecast.steps.data_processing import (
    load_data,
    preprocess_data,
    visualize_sales_data,
)
from zenml_ml_forecast.steps.predictor import generate_forecasts_from_api_step

k8s_settings = KubernetesOrchestratorSettings(
    orchestrator_pod_settings=KubernetesPodSettings(
        resources={
            "requests": {
                "cpu": "1",
                "memory": "2Gi"
            },
            "limits": {
                "cpu": "2",
                "memory": "4Gi"
            }
        }
    ),
    service_account_name="zenml-service-account"
)

load_dotenv()

docker_settings = DockerSettings(
    environment={
        "CLOUD_RUN_API_PREDICT_ENDPOINT": os.getenv("CLOUD_RUN_API_PREDICT_ENDPOINT"),
        "CLOUD_RUN_API_URL": os.getenv("CLOUD_RUN_API_URL")
    }   
)
@pipeline(
    name="ml_forecast_inference_pipeline",
    settings={
        "orchestrator": k8s_settings
    }
)
def inference_pipeline():
    """Pipeline to make retail demand forecasts using trained Prophet models.

    This pipeline is for when you already have trained models and want to
    generate new forecasts without retraining.

    Steps:
    1. Load sales data
    2. Preprocess data
    3. Generate forecasts using provided models or simple baseline models

    Returns:
        combined_forecast: Combined dataframe with all series forecasts
        forecast_dashboard: HTML dashboard with forecast visualizations
        sales_visualization: Interactive visualization of historical sales patterns
    """
    # Load data
    sales_data = load_data()

    # Preprocess data
    train_data_dict, test_data_dict, series_ids = preprocess_data(
        sales_data=sales_data,
        test_size=0.05,  # Just a small test set for visualization purposes
    )

    # Create interactive visualizations of historical sales patterns
    sales_viz = visualize_sales_data(
        sales_data=sales_data,
        train_data_dict=train_data_dict,
        test_data_dict=test_data_dict,
        series_ids=series_ids,
    )

    # Generate forecasts
    _, combined_forecast, forecast_dashboard = generate_forecasts_from_api_step(
        train_data_dict=train_data_dict,
        series_ids=series_ids,
    )

    # Return forecast data and dashboard
    return combined_forecast, forecast_dashboard, sales_viz

def main():
    """Run a simplified retail forecasting pipeline with ZenML.
    """
    pipeline_options = {}
   
    config_path = Path("zenml_ml_forecast/configs/inference.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Set config path
    pipeline_options["config_path"] = config_path

    logger.info("\n" + "=" * 80)
    logger.info(f"Using configuration from: {config_path}")

    logger.info("Running retail forecasting inference pipeline...")
    run = inference_pipeline.with_options(**pipeline_options)()
    
    logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed successfully! Run ID: {run.id}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
