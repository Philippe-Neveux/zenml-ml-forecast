import os
import bentoml
import mlflow

import pandas as pd
from typing import Literal

# Define the runtime environment for your Bento
demo_image = bentoml.images.Image(python_version="3.10") \
    .python_packages(
        "bentoml==1.4.24",
        "mlflow==2.22.2",
        "prophet==1.1.7"
    )

SEGMENTS = [
    "Store_1-Item_A", "Store_1-Item_B", "Store_1-Item_C", "Store_1-Item_D", "Store_1-Item_E",
    "Store_2-Item_A", "Store_2-Item_B", "Store_2-Item_C", "Store_2-Item_D", "Store_2-Item_E",
    "Store_3-Item_A", "Store_3-Item_B", "Store_3-Item_C", "Store_3-Item_D", "Store_3-Item_E"
]

SegmentType = Literal[
    "Store_1-Item_A", "Store_1-Item_B", "Store_1-Item_C", "Store_1-Item_D", "Store_1-Item_E",
    "Store_2-Item_A", "Store_2-Item_B", "Store_2-Item_C", "Store_2-Item_D", "Store_2-Item_E",
    "Store_3-Item_A", "Store_3-Item_B", "Store_3-Item_C", "Store_3-Item_D", "Store_3-Item_E"
]

@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class ProphetModel:
    segments = SEGMENTS
    prophet_model_names = [f"prophet_model_{segment}" for segment in segments]

    def __init__(self):
        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        else:
            raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")
        self.models = {}
        
        # Load models from MLflow Model Registry
        for model_name in self.prophet_model_names:
            self.models[model_name] = mlflow.prophet.load_model(
                f"models:/{model_name}/latest"
            )

    # Define an API endpoint
    @bentoml.api()
    def predict(
        self,
        segment: SegmentType,
        period: int = 365
    ) -> pd.DataFrame:
        future = self.models[segment].make_future_dataframe(periods=period)
        forecast = self.models[segment].predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(period)
    
    @bentoml.api()
    def health(
        self
    ) -> str:
        return "healthy"