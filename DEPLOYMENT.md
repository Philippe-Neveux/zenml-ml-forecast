# GitHub Actions CD Workflow Configuration

This document describes the required secrets and environment variables needed for the CD workflow to deploy your BentoML service to Google Cloud Run.

## Required GitHub Secrets

You need to configure the following secrets in your GitHub repository settings (Settings > Secrets and variables > Actions):

### 1. `GCP_PROJECT_ID` (Required)
- **Description**: Your Google Cloud Project ID
- **Example**: `my-gcp-project-123456`
- **How to get**: Found in your GCP Console dashboard

### 2. `GCP_SA_KEY` (Required)
- **Description**: JSON key for a Google Cloud Service Account with necessary permissions
- **Format**: Complete JSON content of the service account key file
- **Required permissions**:
  - Artifact Registry Administrator
  - Cloud Run Admin
  - Service Account User
  - Storage Admin (if using GCS for model storage)

#### Creating the Service Account:
```bash
# Create service account
gcloud iam service-accounts create bentoml-deployer \
    --description="Service account for BentoML deployment" \
    --display-name="BentoML Deployer"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:bentoml-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:bentoml-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:bentoml-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Create and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=bentoml-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 3. `MLFLOW_TRACKING_URI` (Optional)
- **Description**: MLflow tracking server URI
- **Default**: `https://mlflow.34.40.165.212.nip.io`
- **Example**: `https://your-mlflow-server.com`

## Optional GitHub Variables

You can configure these in Settings > Secrets and variables > Actions > Variables tab:

### 1. `GAR_LOCATION`
- **Description**: Google Artifact Registry location
- **Default**: `us-central1`
- **Example**: `europe-west1`

### 2. `GAR_REPOSITORY`
- **Description**: Artifact Registry repository name
- **Default**: `zenml-ml-forecast`
- **Example**: `my-ml-models`

### 3. `CLOUD_RUN_SERVICE`
- **Description**: Cloud Run service name
- **Default**: `prophet-forecasting-service`
- **Example**: `ml-forecasting-api`

### 4. `CLOUD_RUN_REGION`
- **Description**: Cloud Run deployment region
- **Default**: `us-central1`
- **Example**: `europe-west1`

## Workflow Triggers

The workflow is triggered by:
1. **Push to main branch** with changes to:
   - `zenml_ml_forecast/**`
   - `bentofile.yaml`
   - `pyproject.toml`
2. **Manual trigger** via GitHub Actions UI (`workflow_dispatch`)

## Deployment Configuration

The Cloud Run service will be deployed with:
- **Memory**: 2Gi
- **CPU**: 2 cores
- **Port**: 3000
- **Timeout**: 300 seconds
- **Concurrency**: 1000 requests
- **Scaling**: 0-10 instances
- **Authentication**: Allow unauthenticated requests

## Testing the Deployment

After deployment, the workflow will:
1. Display the service URL
2. Perform a basic health check
3. Test the prediction endpoint with sample data

## Manual Setup Steps

Before running the workflow for the first time:

1. Enable required GCP APIs:
```bash
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

2. Set up your GitHub repository secrets as described above

3. Ensure your MLflow models are accessible from the specified tracking URI

4. Trigger the workflow by pushing to main or using manual dispatch

## Troubleshooting

- **Authentication errors**: Verify your service account key and permissions
- **Build failures**: Check that all dependencies are properly specified in `bentofile.yaml`
- **Deployment failures**: Verify Cloud Run quotas and region availability
- **Model loading errors**: Ensure MLflow tracking URI is accessible and models exist