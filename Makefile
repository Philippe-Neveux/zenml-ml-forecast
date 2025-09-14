zenml-login:
	uv run zenml login https://zenml-server.34.40.173.65.nip.io

zenml-init:
	uv run zenml init

BUCKET_NAME := zenml-zenml-artifacts
zenml-register-artifact-store:
	@echo "Registering GCS artifact store in ZenML..."
	uv run zenml artifact-store register gs_store_ml_forecast \
		-f gcp \
		--path=gs://$(BUCKET_NAME)/ml_forecast/

zenml-register-code-repository:
	@echo "Registering code repository in ZenML..."
	source .env && uv run zenml code-repository register github_ml_forecast \
		--type=github \
		--owner=Philippe-Neveux \
		--repository=zenml-ml-forecast

zenml-configure-mlops-stack:
	@echo "Configure MLOps stack with each component ..."
	uv run zenml stack register ml_forecast_stack \
		-a gs_store_ml_forecast \
		-c zenml-artifact-registry \
		-o kubernetes_orchestrator \
		-e mlflow \
		-r mlflow_model_registry \
		--set


# Kubernetes
connect-k8s-cluster:
	echo "Connecting to Kubernetes cluster..."
	gcloud container clusters get-credentials zenml --region australia-southeast1 --project zenml-470505


# Run Pipeline
run-main-pipeline:
	uv run python src/zenml_ml_forecast/main.py