from pathlib import Path
from typing import Dict, Tuple

from loguru import logger
import mlflow
from typing_extensions import Annotated
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.kubernetes.flavors import KubernetesOrchestratorSettings
from zenml.integrations.kubernetes.pod_settings import KubernetesPodSettings
from zenml.types import HTMLString

from zenml_ml_forecast.steps.data_processing import (
    load_data,
    preprocess_data,
    visualize_sales_data,
)
from zenml_ml_forecast.steps.model import evaluate_models, train_model
from zenml_ml_forecast.steps.predictor import generate_forecasts

docker_settings = DockerSettings(
    target_repository="zenml-ml-forecast"
)

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
        },
        # Install the package in the container
        post_commands=[
            "pip install -e /app/code"
        ]
    ),
    service_account_name="zenml-service-account"
)

@pipeline(
    name="ml_forecast_training_pipeline",
    settings={
        "docker": docker_settings,
        "orchestrator": k8s_settings
    }
)
def training_pipeline() -> Tuple[
    Annotated[Dict[str, float], "model_metrics"],
    Annotated[HTMLString, "evaluation_report"],
    Annotated[HTMLString, "forecast_dashboard"],
    Annotated[HTMLString, "sales_visualization"],
]:
    """Simple retail forecasting pipeline using Prophet.

    Steps:
    1. Load sales data
    2. Preprocess data for Prophet
    3. Visualize historical sales patterns (interactive)
    4. Train Prophet models (one per store-item combination)
    5. Evaluate model performance on test data
    6. Generate forecasts for future periods

    Args:
        test_size: Proportion of data to use for testing
        forecast_periods: Number of days to forecast into the future
        weekly_seasonality: Whether to include weekly seasonality in the model

    Returns:
        model_metrics: Dictionary of performance metrics
        evaluation_report: HTML report of model evaluation
        forecast_dashboard: HTML dashboard of forecasts
        sales_visualization: Interactive visualization of historical sales patterns
    """
    # Load synthetic retail data
    sales_data = load_data()

    # Preprocess data for Prophet
    train_data_dict, test_data_dict, series_ids = preprocess_data(
        sales_data=sales_data
    )

    # Create interactive visualizations of historical sales patterns
    sales_viz = visualize_sales_data(
        sales_data=sales_data,
        train_data_dict=train_data_dict,
        test_data_dict=test_data_dict,
        series_ids=series_ids,
    )

    # Train Prophet models for each series
    models = train_model(
        train_data_dict=train_data_dict,
        series_ids=series_ids,
    )
    
    #Log all models in mlflow
    for series_id, model in models.items():
        logger.info(f"Logging model for series_id: {series_id}")
        model.log_model(
            artifact_path=f"prophet_model_{series_id}",
            registered_model_name=f"prophet_model_{series_id}"
        )
        mlflow.prophet.log_model(
            pr_model=model,
            name=series_id,
            input_example=train_data_dict[series_id].head()
        )

    # Evaluate models
    metrics, evaluation_report = evaluate_models(
        models=models, test_data_dict=test_data_dict, series_ids=series_ids
    )

    # Generate forecasts
    _, _, forecast_dashboard = generate_forecasts(
        models=models,
        train_data_dict=train_data_dict,
        series_ids=series_ids,
    )

    return metrics, evaluation_report, forecast_dashboard, sales_viz


def main(
):
    """Run a simplified retail forecasting pipeline with ZenML.

    Args:
        config: Path to the configuration YAML file
        no_cache: Disable caching for the pipeline run
    """
    pipeline_options = {}
   
    config_path = Path("zenml_ml_forecast/configs/training.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Set config path
    pipeline_options["config_path"] = config_path

    logger.info("\n" + "=" * 80)
    logger.info(f"Using configuration from: {config_path}")

    logger.info("Running retail forecasting training pipeline...")
    run = training_pipeline.with_options(**pipeline_options)()

    logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed successfully! Run ID: {run.id}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
