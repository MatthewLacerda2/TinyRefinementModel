This model is meant to prove latent reasoning beats token-level reasoning.
We use a scratchpad to store the intermediate states of the reasoning process.

This is a very short readme file

I have a GCP bucket to store the pretokenized dataset (look prefill.py) and the model's weights

The model is trained in a v5litepod-4 VM.
The start of the VM, the training process and the upload of the model's weights to the bucket are all done manually. Training the model is not something you do for cheap, thus not often and thus should be monitored live.
