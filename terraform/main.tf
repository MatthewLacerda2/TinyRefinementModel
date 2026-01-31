provider "google" {
  project = "your-project-id"
  region  = "us-central1"
}

# 1. The Storage Bucket for Checkpoints (Orbax)
resource "google_storage_bucket" "model_bucket" {
  name     = "recursive-beast-checkpoints"
  location = "US"
  force_destroy = true # Good for research, dangerous for production
}

# 2. The TPU Node (Spot/Preemptible for 70-90% discount)
resource "google_tpu_v2_node" "tpu_beast" {
  name = "tpu-beast-01"
  zone = "us-central1-c"
  accelerator_config {
    type     = "V2_8"
    topology = "2x2"
  }
  runtime_version = "tpu-vm-v2-base" # JAX-native
  
  # The Money Saver: Spot Instance
  scheduling_config {
    preemptible = true
  }

  network_config {
    network = "default"
    subnetwork = "default"
  }
}

output "tpu_ip" {
  value = google_tpu_v2_node.tpu_beast.network_endpoints[0].ip_address
}