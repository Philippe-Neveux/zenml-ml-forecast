from loguru import logger
from zenml import get_pipeline_context, pipeline

from zenml_ml_forecast.steps.data_loader import load_data
from zenml_ml_forecast.steps.data_preprocessor import preprocess_data
from zenml_ml_forecast.steps.data_visualizer import visualize_sales_data
from zenml_ml_forecast.steps.predictor import generate_forecasts


@pipeline(name="retail_forecast_inference_pipeline")
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

    # Get the models from the Model Registry
    models = get_pipeline_context().model.get_artifact(
        "trained_prophet_models"
    )

    # Generate forecasts
    _, combined_forecast, forecast_dashboard = generate_forecasts(
        models=models,
        train_data_dict=train_data_dict,
        series_ids=series_ids,
    )

    # Return forecast data and dashboard
    return combined_forecast, forecast_dashboard, sales_viz


@click.command(
    help="""
RetailForecast - Simple Retail Demand Forecasting with ZenML and Prophet

Run a simplified retail demand forecasting pipeline using Facebook Prophet.

Examples:

  \b
  # Run the training pipeline with default training config
  python run.py

  \b
  # Run with a specific training configuration file
  python run.py --config configs/training.yaml
  
  \b
  # Run the inference pipeline with default inference config
  python run.py --inference
"""
)
@click.option(
    "--config",
    type=str,
    default=None,
    help="Path to the configuration YAML file",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run",
)
@click.option(
    "--inference",
    is_flag=True,
    default=False,
    help="Run the inference pipeline instead of the training pipeline",
)
@click.option(
    "--log-file",
    type=str,
    default=None,
    help="Path to log file (if not provided, logs only go to console)",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging",
)
def main(
    config: str = None,
    no_cache: bool = False,
    inference: bool = False,
    log_file: str = None,
    debug: bool = False,
):
    """Run a simplified retail forecasting pipeline with ZenML.

    Args:
        config: Path to the configuration YAML file
        no_cache: Disable caching for the pipeline run
        inference: Run the inference pipeline instead of the training pipeline
        log_file: Path to log file
        debug: Enable debug logging
    """
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    configure_logging(level=log_level, log_file=log_file)

    pipeline_options = {}
    if no_cache:
        pipeline_options["enable_cache"] = False

    # Select default config based on pipeline type if not specified
    if config is None:
        config = (
            "configs/inference.yaml" if inference else "configs/training.yaml"
        )

    # Set config path
    pipeline_options["config_path"] = config

    logger.info("\n" + "=" * 80)
    logger.info(f"Using configuration from: {config}")

    # Run the appropriate pipeline
    if inference:
        logger.info("Running retail forecasting inference pipeline...")
        run = inference_pipeline.with_options(**pipeline_options)()
    else:
        logger.info("Running retail forecasting training pipeline...")
        run = training_pipeline.with_options(**pipeline_options)()

    logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed successfully! Run ID: {run.id}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
