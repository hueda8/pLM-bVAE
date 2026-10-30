# Step 1: Train pLM-bVAE

This directory contains the pLM-bVAE training and evaluation workflow. The example in [`esm_650m_opt/`](esm_650m_opt/) uses residue-level representations from ESM-2 650M to encode sequences as binary vectors and reconstruct amino acid sequences.

## Procedure

1. Prepare the sequence files and CDRH3 residue indices in [step 0](../step0_data_preparation/) and set up the [pLM-bVAE and MOQA environment](../environments/plm_bvae_moqa/README.md).
2. In `esm_650m_opt/experiment_configs/binary.yml`, set the data, embeddings, model, and output paths. Adapt the shell scripts and `set_envs/` to the target machine. Downloading ESM-2 requires network access; GPU-enabled PyTorch is recommended.
3. From `esm_650m_opt/`, run:

   ```bash
   bash emb_esm2.sh
   bash train_bvae.sh
   bash eval_bvae.sh
   bash encode_bvae.sh
   bash decode_bvae.sh
   ```

The scripts generate embeddings for the training, validation, and test sets; train the model; evaluate it on the test set; encode test sequences into binary vectors; and reconstruct sequences from those vectors. Embeddings are stored according to `embeddings_path`; model checkpoints and results are stored under the configured `model_path` and `output_path`.

For optional hyperparameter optimization, configure and run `cre_opt_db.sh` and then `opt_bvae.sh`. Select and apply the resulting hyperparameters before training. The `eval_comp_bvae.sh` comparison script also contains paths and an environment switch that must be adapted.

Use the trained `best/model.pt` checkpoint and matching configuration in [step 3](../step3_run_moqa/). The environment is a candidate and should be validated on the target machine.

[Tutorial overview](../README.md)
