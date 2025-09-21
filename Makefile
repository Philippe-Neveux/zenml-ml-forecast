INGRESS_IP_ADDRESS := 34.40.165.212

https://zenml-server.34.40.165.212.nip.io

zenml-login:
	uv run zenml login https://zenml-server.$(INGRESS_IP_ADDRESS).nip.io

zenml-init:
	uv run zenml init

BUCKET_NAME := zenml-472221-zenml-artifacts
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
		--repository=zenml-ml-forecast \
		--token=$$GITHUB_TOKEN

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
	gcloud container clusters get-credentials zenml \
		--region australia-southeast1 \
		--project zenml-472221

kubectl-set-namespace-zenml:
	@echo "Setting default namespace to zenml..."
	kubectl config set-context --current --namespace=zenml

get-pods: kubectl-set-namespace-zenml
	@echo "Listing pods in the current namespace..."
	kubectl get pods --sort-by=.metadata.creationTimestamp

kubectl-cleanup-completed-pods:
	@echo "Removing completed pods in all namespaces..."
	# Remove succeeded pods
	kubectl delete pods --field-selector=status.phase=Succeeded
	# Remove failed pods  
	kubectl delete pods --field-selector=status.phase=Failed

# Run Pipeline
gcp-connect-to-artifact-registry:
	@echo "Connecting GCP to Artifact Registry..."
	gcloud auth configure-docker australia-southeast1-docker.pkg.dev --quiet

run-main-pipeline: gcp-connect-to-artifact-registry
	@echo "Running the main pipeline..."
	uv run main

run-training-pipeline: gcp-connect-to-artifact-registry
	@echo "Running the training pipeline..."
	uv run training

run-inference-pipeline: gcp-connect-to-artifact-registry
	@echo "Running the inference pipeline..."
	uv run inference

ruff:
	uv run ruff check zenml_ml_forecast --fix --select I

API_URL := https://prophet-forecasting-service-883174852290.australia-southeast1.run.app
# API deployed
test-api:
	curl -X POST "$(API_URL)/predict" \
	-H "Content-Type: application/json" \
	-H "Authorization: Bearer $(shell gcloud auth print-identity-token)" \
	-d '{"segment": "prophet_model_Store_1-Item_A", "period": 365}'

check-logs-api:
	@echo "Checking logs for the Cloud Run service..."
	gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=prophet-forecasting-service" \
		--project=zenml-472221 \
		--limit=50 \
		--format="value(textPayload)"