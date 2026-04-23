output "project_number" {
  value       = data.google_project.project.number
  description = "The GCP Project Number"
}

output "bucket_name" {
  value       = google_storage_bucket.model_data.name
  description = "The name of the created GCS bucket"
}

output "bucket_url" {
  value       = google_storage_bucket.model_data.url
  description = "The base URL of the bucket"
}
