This model is meant to prove latent reasoning beats token-level reasoning.
We use a scratchpad to store the intermediate states of the reasoning process.

I won't bother explaining more because i change the code too frequently.

# Infra

I have a terraform setup so we can train this on TPU Spot Instances on Google Cloud.

There is a setup for a short run just to prove the architecture and another setup for training a full model.

At step 2100, i stopped and resumed with ultrachat only