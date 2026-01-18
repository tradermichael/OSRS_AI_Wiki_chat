param(
  [Parameter(Mandatory=$true)][string]$ProjectId,
  [string]$Region = "us-central1",
  [string]$ServiceName = "osrs-ai-wiki-chat",
  [string]$ArtifactRepo = "osrs-ai-wiki-chat",
  [string]$GitHubOwner = "tradermichael",
  [string]$GitHubRepo = "OSRS_AI_Wiki_chat",
  [string]$VertexLocation = "us-central1"
)

$ErrorActionPreference = "Stop"

Write-Host "Setting project to $ProjectId"
gcloud config set project $ProjectId

Write-Host "Enabling required APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com iamcredentials.googleapis.com sts.googleapis.com

Write-Host "Creating Artifact Registry repo (if missing)"
gcloud artifacts repositories describe $ArtifactRepo --location $Region | Out-Null
if ($LASTEXITCODE -ne 0) {
  gcloud artifacts repositories create $ArtifactRepo --repository-format=docker --location=$Region --description="OSRS AI Wiki Chat images"
}

$SaName = "$ServiceName-deployer"
$SaEmail = "$SaName@$ProjectId.iam.gserviceaccount.com"

$RuntimeSaName = "$ServiceName-runtime"
$RuntimeSaEmail = "$RuntimeSaName@$ProjectId.iam.gserviceaccount.com"

Write-Host "Creating service account (if missing): $SaEmail"
gcloud iam service-accounts describe $SaEmail | Out-Null
if ($LASTEXITCODE -ne 0) {
  gcloud iam service-accounts create $SaName --display-name "$ServiceName GitHub deployer"
}

Write-Host "Creating runtime service account (if missing): $RuntimeSaEmail"
gcloud iam service-accounts describe $RuntimeSaEmail | Out-Null
if ($LASTEXITCODE -ne 0) {
  gcloud iam service-accounts create $RuntimeSaName --display-name "$ServiceName Cloud Run runtime"
}

Write-Host "Granting IAM roles"
$roles = @(
  "roles/run.admin",
  "roles/artifactregistry.writer",
  "roles/iam.serviceAccountUser"
)
foreach ($role in $roles) {
  gcloud projects add-iam-policy-binding $ProjectId --member "serviceAccount:$SaEmail" --role $role | Out-Null
}

Write-Host "Granting runtime permissions (Vertex AI)"
gcloud projects add-iam-policy-binding $ProjectId --member "serviceAccount:$RuntimeSaEmail" --role roles/aiplatform.user | Out-Null

Write-Host "Creating Workload Identity Pool + Provider"
$poolId = "$ServiceName-pool"
$providerId = "$ServiceName-provider"

gcloud iam workload-identity-pools describe $poolId --location=global | Out-Null
if ($LASTEXITCODE -ne 0) {
  gcloud iam workload-identity-pools create $poolId --location=global --display-name "OSRS Chat GH Pool"
}

$issuer = "https://token.actions.githubusercontent.com"

gcloud iam workload-identity-pools providers describe $providerId --location=global --workload-identity-pool=$poolId | Out-Null
if ($LASTEXITCODE -ne 0) {
  gcloud iam workload-identity-pools providers create-oidc $providerId `
    --location=global `
    --workload-identity-pool=$poolId `
    --display-name "OSRS Chat GH Provider" `
    --issuer-uri $issuer `
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" `
    --attribute-condition "assertion.repository_owner=='$GitHubOwner' && assertion.repository=='$GitHubOwner/$GitHubRepo'"
}

$providerResource = "projects/$(gcloud projects describe $ProjectId --format='value(projectNumber)')/locations/global/workloadIdentityPools/$poolId/providers/$providerId"

Write-Host "Binding WIF principal to service account"
$principal = "principalSet://iam.googleapis.com/projects/$(gcloud projects describe $ProjectId --format='value(projectNumber)')/locations/global/workloadIdentityPools/$poolId/attribute.repository/$GitHubOwner/$GitHubRepo"

gcloud iam service-accounts add-iam-policy-binding $SaEmail `
  --role roles/iam.workloadIdentityUser `
  --member $principal | Out-Null

Write-Host "Bootstrap complete. Add these GitHub secrets:"
Write-Host "  GCP_PROJECT_ID=$ProjectId"
Write-Host "  GCP_REGION=$Region"
Write-Host "  GCP_VERTEX_LOCATION=$VertexLocation"
Write-Host "  GCP_ARTIFACT_REPO=$ArtifactRepo"
Write-Host "  GCP_SERVICE_ACCOUNT_EMAIL=$SaEmail"
Write-Host "  GCP_RUNTIME_SERVICE_ACCOUNT_EMAIL=$RuntimeSaEmail"
Write-Host "  GCP_WORKLOAD_IDENTITY_PROVIDER=$providerResource"
