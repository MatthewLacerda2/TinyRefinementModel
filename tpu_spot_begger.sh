#!/bin/bash

# Configuration
PROJECT_ID="king-mark"
TPU_NAME="reasoner-tpu"
ACCELERATOR_TYPE="v5litepod-8"
RUNTIME_VERSION="v2-alp1ha-tpuv5-lite" # Optimized for JAX on v5e

# Prioritize us-central1 for higher chance of Spot grantace and 0-cost egress
ZONES=("us-central1-a" "us-central1-b" "us-south1-a" "us-west1-c")

echo "🎯 Starting the hunt. Data bucket is in us-central1."

while true; do
  for ZONE in "${ZONES[@]}"; do
    QR_ID="req-$(date +%s)"
    echo "🎲 Attempting $ZONE..."

    # We use Queued Resources to 'line up' for the spot capacity
    gcloud compute tpus queued-resources create $QR_ID \
      --node-id=$TPU_NAME \
      --zone=$ZONE \
      --accelerator-type=$ACCELERATOR_TYPE \
      --runtime-version=$RUNTIME_VERSION \
      --project=$PROJECT_ID \
      --spot --async > /dev/null 2>&1

    if [ $? -eq 0 ]; then
      echo "📥 Request submitted. Waiting for provisioning in $ZONE..."
      
      # Check status every 15 seconds
      while true; do
        STATE=$(gcloud compute tpus queued-resources describe $QR_ID --zone=$ZONE --format="value(state)")
        
        if [ "$STATE" == "ACTIVE" ]; then
          echo "✅ TPU IS ALIVE! Zone: $ZONE"
          exit 0
        elif [ "$STATE" == "FAILED" ]; then
          echo "❌ Stockout in $ZONE. Cleaning up request..."
          gcloud compute tpus queued-resources delete $QR_ID --zone=$ZONE --quiet --async
          break 
        fi
        echo -n "."
        sleep 15
      done
    fi
  done
  echo -e "\n💤 No capacity in preferred zones. Retrying in 30s..."
  sleep 30
done