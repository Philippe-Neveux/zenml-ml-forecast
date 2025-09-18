import bentoml
import mlflow
import numpy as np

import pandas as pd
from bentoml.models import BentoModel

# Define the runtime environment for your Bento
demo_image = bentoml.images.Image(python_version="3.10") \
    .python_packages("mlflow", "prophet")


@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class ProphetModel:
    # Declare the model as a class attribute
    # bento_model = BentoModel("prophet_model_Store_1-Item_A:latest")

    def __init__(self):
        mlflow.set_tracking_uri("https://mlflow.34.40.165.212.nip.io")
        self.model = mlflow.prophet.load_model(
            "models:/prophet_model_Store_1-Item_A/latest"
        )

    # Define an API endpoint
    @bentoml.api()
    def predict(self, period: int = 365) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=period)
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(period)