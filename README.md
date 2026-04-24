This model is meant to prove latent reasoning beats token-level reasoning.
We use a scratchpad to store the intermediate states of the reasoning process.

# How to train

_Note: once you trained the model and saved it's weights, you can run it in any bfloat16 compatible machine._

You must have 'gcloud' installed.

Run `prefill.py` to pretokenize the dataset.

Create a project in Google Cloud.

Copy `terraform.tfvars.example` to `terraform.tfvars`, replace the values, and run `terraform init && terraform apply`.

Run `gcloud storage cp -r tpu_data gs://<your-bucket>/` to copy the pretokenized dataset to the cloud bucket.
(This can take a while, read the rest of the README.md in the meantime)

Create a `v5litepod-4` VM.
_This part you'll do manually. Training is sporadic, costly and finnicky, and should be monitored live._
Go to the Cloud Console -> AI Platform -> TPU VMs -> Create VM
Use the following values:
- Name: `my-name`
- Region: `us-central1`
- Type: `v5litepod-4`
- Runtime version: `v2-alpha-tpuv5-lite`
- TPU VM: True

Once the machine is created:

1. SSH into the machine and `git clone` this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Create a `.env` file and set the data source:
   ```bash
   echo "DATA_ROOT=gs://<your-bucket>/tpu_data" > .env
   ```
4. Start training: `python start_training.py`.

You can play with the model using `python infer_local.py`.
Don't stick around too long, the VM is $5.00 USD/hour

5. After training, copy your checkpoints back to the bucket:
   ```bash
   gcloud storage cp -r orbax_checkpoints gs://<your-bucket>/

   #Also copy the training logs and plots
   python plot_history.py
   gcloud storage cp -r training_history.csv gs://<your-bucket>/
   gcloud storage cp -r reasoning_analytics.png gs://<your-bucket>/
   ```

6. Don't forget to shut down your machine to stop the billing:
   ```bash
   sudo shutdown now
   ```