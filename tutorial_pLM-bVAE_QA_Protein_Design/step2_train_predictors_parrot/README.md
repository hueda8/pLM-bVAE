# Step 2: Train the PARROT predictors

This directory trains three PARROT predictors to estimate objective function values for CDRH3 variant sequences. Their predictions are used separately in the MOQA workflow.

## Procedure

1. Prepare the sequence-score pairs and split indices in [step 0](../step0_data_preparation/). Set up the [PARROT environment](../environments/parrot/README.md), including its modified training files.
2. Update the data and split paths in `model1/train.sh`, `model2/train.sh`, and `model3/train.sh`. The prepared index file is named `cdrh3_random_indices.txt`, whereas the scripts currently reference `cdrh3_indices.txt`. Check the path to [`zscale3_encoding.txt`](../zscale3_encoding.txt) and adapt the environment setup in `set_envs/` to the target machine.
3. Run each `train.sh` from its own model directory. Each script writes a trained predictor to `model/network.pt`.

The optional [`model_optimize/opt_model.sh`](model_optimize/opt_model.sh) runs predictor hyperparameter optimization on the combined training and validation data; adjust its paths before use. For [step 3](../step3_run_moqa/), supply the three trained predictor files and the same amino acid descriptor encoding used for training.

[Tutorial overview](../README.md)
