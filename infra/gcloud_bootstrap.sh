#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-osrs-ai-wiki-chat}"
ARTIFACT_REPO="${ARTIFACT_REPO:-osrs-ai-wiki-chat}"
GITHUB_OWNER="${GITHUB_OWNER:-tradermichael}"
GITHUB_REPO="${GITHUB_REPO:-OSRS_AI_Wiki_chat}"


gcloud config set project "$PROJECT_ID"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  firestore.googleapis.com \
  datastore.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com

if ! gcloud artifacts repositories describe "$ARTIFACT_REPO" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$ARTIFACT_REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="OSRS AI Wiki Chat images"
fi

SA_NAME="$SERVICE_NAME-deployer"
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

RUNTIME_SA_NAME="$SERVICE_NAME-runtime"
RUNTIME_SA_EMAIL="$RUNTIME_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" --display-name "$SERVICE_NAME GitHub deployer"
fi

if ! gcloud iam service-accounts describe "$RUNTIME_SA_EMAIL" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$RUNTIME_SA_NAME" --display-name "$SERVICE_NAME Cloud Run runtime"
fi

for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$SA_EMAIL" \
    --role "$role" >/dev/null
done

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:$RUNTIME_SA_EMAIL" \
  --role roles/aiplatform.user >/dev/null

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:$RUNTIME_SA_EMAIL" \
  --role roles/datastore.user >/dev/null

POOL_ID="$SERVICE_NAME-pool"
PROVIDER_ID="$SERVICE_NAME-provider"

if ! gcloud iam workload-identity-pools describe "$POOL_ID" --location=global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL_ID" --location=global --display-name "OSRS Chat GH Pool"
fi

ISSUER="https://token.actions.githubusercontent.com"

if ! gcloud iam workload-identity-pools providers describe "$PROVIDER_ID" --location=global --workload-identity-pool="$POOL_ID" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_ID" \
    --location=global \
    --workload-identity-pool="$POOL_ID" \
    --display-name "OSRS Chat GH Provider" \
    --issuer-uri "$ISSUER" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
    --attribute-condition "assertion.repository_owner=='$GITHUB_OWNER' && assertion.repository=='$GITHUB_OWNER/$GITHUB_REPO'"
fi

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
PROVIDER_RESOURCE="projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL_ID/providers/$PROVIDER_ID"

PRINCIPAL="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL_ID/attribute.repository/$GITHUB_OWNER/$GITHUB_REPO"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role roles/iam.workloadIdentityUser \
  --member "$PRINCIPAL" >/dev/null

echo "Bootstrap complete. Add GitHub secrets:"
echo "  GCP_PROJECT_ID=$PROJECT_ID"
echo "  GCP_REGION=$REGION"
echo "  GCP_ARTIFACT_REPO=$ARTIFACT_REPO"
echo "  GCP_SERVICE_ACCOUNT_EMAIL=$SA_EMAIL"
echo "  GCP_RUNTIME_SERVICE_ACCOUNT_EMAIL=$RUNTIME_SA_EMAIL"
echo "  GCP_WORKLOAD_IDENTITY_PROVIDER=$PROVIDER_RESOURCE"
