# Step 3: Run MOQA

This directory contains the multi-objective optimization by quantum annealing (MOQA) workflow. The sampler decodes binary vectors with pLM-bVAE, evaluates the resulting sequences with three PARROT predictors, and iteratively samples binary vectors using a factorization machine and a quantum annealer.

## Prepare the inputs

Set up the [pLM-bVAE and MOQA environment](../environments/plm_bvae_moqa/README.md), including the modified PARROT predictor. In [`esm_650m_opt_moqa1/`](esm_650m_opt_moqa1/):

1. Set `model_path` in `experiment_configs/binary.yml` to the pLM-bVAE model directory from [step 1](../step1_train_plm_bvae/) containing `best/model.pt`. Use a matching configuration and set `output_path` consistently with the sampler's `model_output/binary` paths.
2. Place the three predictor weights from [step 2](../step2_train_predictors_parrot/) at `parrot_models/model_network_1.pt`, `model_network_2.pt`, and `model_network_3.pt`, or change their paths in `sampler_moqa.py`. Check the path to [`zscale3_encoding.txt`](../zscale3_encoding.txt).
3. Place initial binary vectors in `model_output/binary/vectors.txt`. Example 64-dimensional sets are available in [`../data/moqa_initial_data/`](../data/moqa_initial_data/); their dimension must match the trained model.
4. Configure the objective direction, hypervolume reference point, D-Wave endpoint, solver, and credentials in `sampler_moqa.py`. Access to a compatible D-Wave solver is required. Keep credentials out of files committed to a public repository.

## Run the sampler

Adapt `run_moqa.sh` and the environment setup to the target machine. The script currently refers to an older setup script; change it to use `setenv_plm-bvae-moqa-py310.sh`. Then run from `esm_650m_opt_moqa1/`:

```bash
bash run_moqa.sh
```

The sampler writes generated vectors, reconstructed sequences, and optimization logs under `model_output/binary/`. It can remove or overwrite existing output files, so preserve results before starting another run. Validate a small run on the target machine before a full experiment.

[Tutorial overview](../README.md)
