# We use a data source to reference the existing project without needing to fully manage 
# its lifecycle (which would require you to provide billing/org IDs).
data "google_project" "project" {
  project_id = var.project_id
}

# Optional: Ensure the required APIs are enabled for your project.
resource "google_project_service" "storage_api" {
  project            = data.google_project.project.project_id
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

# Optional: Enable Compute Engine API as you plan to spin up VMs manually
resource "google_project_service" "compute_api" {
  project            = data.google_project.project.project_id
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

# The GCS bucket to store the pretokenized dataset and model weights.
resource "google_storage_bucket" "model_data" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = false

  # Standard storage is best for data accessed frequently (like training datasets and model checkpoints)
  storage_class = "STANDARD"

  # Uniform bucket-level access is highly recommended for security and simpler IAM management
  uniform_bucket_level_access = true

  # Optional: Automatically abort incomplete multi-part uploads to save costs
  lifecycle_rule {
    action {
      type = "AbortIncompleteMultipartUpload"
    }
    condition {
      age = 7
    }
  }

  depends_on = [
    google_project_service.storage_api
  ]
}
