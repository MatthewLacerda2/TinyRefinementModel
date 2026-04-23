variable "project_id" {
  type        = string
  description = "The GCP Project ID"
}

variable "region" {
  type        = string
  description = "The default GCP region for resources"
  default     = "us-central1"
}

variable "bucket_name" {
  type        = string
  description = "Name of the bucket for dataset and model weights"
}
