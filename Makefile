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
	gcloud container clusters get-credentials zenml --region australia-southeast1 --project zenml-470505

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
run-training-pipeline:
	@echo "Running the training pipeline..."
	uv run training

run-inference-pipeline:
	@echo "Running the inference pipeline..."
	uv run inference

ruff:
	uv run ruff check src/zenml_ml_forecast --fix --select I