# Reproducing the paper

This directory contains the data, scripts, and analysis notebooks for the
paper's experiments on peptide, CDRH3, and AAV variant datasets. The workflow
compares bVAE and protein language model-enhanced bVAE (pLM-bVAE), then uses
the trained models and PARROT predictors for quantum annealing (QA)-based
protein sequence design with MOQA. Existing outputs can also be analyzed
without repeating model training and QA.

## Directory overview

| Directory | Purpose |
| --- | --- |
| [`data/`](data/) | Raw and prepared datasets, PARROT inputs, and initial binary vectors for MOQA. |
| [`data_preparation/`](data_preparation/) | Notebooks for dataset preparation, random splits, and PARROT inputs. |
| [`environments/`](environments/) | Separate environment setup instructions for analysis, bVAE, pLM-bVAE, PARROT, and MOQA. |
| [`set_envs/`](set_envs/) | Shell scripts for activating the environments used by the job scripts. |
| [`train_bvae/`](train_bvae/) | Training, encoding, decoding, and evaluation scripts for bVAE. |
| [`train_plm_bvae/`](train_plm_bvae/) | pLM embedding, pLM-bVAE training, hyperparameter optimization, and evaluation scripts. |
| [`train_predictors_parrot/`](train_predictors_parrot/) | PARROT predictor training and hyperparameter optimization scripts. |
| [`run_moqa/`](run_moqa/) | MOQA samplers for bVAE and pLM-bVAE, including the local `fmqa` module. |
| [`outputs/`](outputs/) | Stored results used by the analysis notebooks, including MOQA and predictor outputs. |
| [`results/`](results/) | Notebooks for the paper's main and supplementary figures and analyses. |

## Reproduction workflow

1. **Set up the environments and paths.** Follow the README for each stage:
   [analysis](environments/analysis/README.md),
   [bVAE](environments/bvae/README.md),
   [pLM-bVAE](environments/plm_bvae/README.md),
   [PARROT](environments/parrot/README.md), and
   [MOQA](environments/moqa/README.md). For PARROT and MOQA, apply the
   replacement PARROT files as described in their respective READMEs. Edit
   the conda paths in `set_envs/`, and replace placeholder paths such as
   `...` in notebooks, shell scripts, and configuration files. Run a shell
   script from its own directory so that its relative paths resolve correctly.

2. **Prepare datasets.** Use the notebooks in `data_preparation/plm_bvae/`
   to prepare the variant sequences and random training, validation, and
   test splits. Use `data_preparation/parrot/` to prepare predictor inputs
   and split indices. The predictors use the same splits as bVAE and
   pLM-bVAE. The CDRH3 preparation notebook downloads source data; check the
   notebook inputs and output paths before running any save cells. Prepared
   inputs are also available under `data/`.

3. **Train and evaluate bVAE and pLM-bVAE.** In each dataset's
   `train_bvae/` directory, configure `experiment_configs/binary.json` and
   use the training and evaluation scripts. In `train_plm_bvae/`, select the
   pLM and experiment configuration, generate the pLM representations with
   the corresponding `emb_*.sh` script, and train and evaluate the model.
   For optimized models, the `*_opt` directories also provide Optuna scripts.
   Several scripts refer to `experiment_configs/binary.yml`; select the
   appropriate model-specific YAML file and provide it under that name or
   update the script. The evaluation scripts use the `best` checkpoint.
   Configure the data, model, and output paths before running.

4. **Train the predictors.** In `train_predictors_parrot/`, configure the
   dataset paths and run the training scripts for the three objective
   functions. The optional `model_optimize/` scripts perform hyperparameter
   optimization. The predictors use the supplied amino acid encoding and the
   matching training and validation splits.

5. **Run QA-based sequence design.** In `run_moqa/`, select the dataset and
   the bVAE or pLM-bVAE sampler. Configure the trained decoder, the three
   trained PARROT predictors, amino acid encoding, initial binary vectors,
   output paths, and D-Wave access. Run `bash run_moqa.sh` from the selected
   run directory. Repeat with the appropriate initial vectors and settings
   for the paper's independent runs. The sampler can overwrite existing
   results; preserve outputs before starting. See the MOQA environment README
   for the additional interpreter issue when using the bVAE decoder.

6. **Analyze the results.** Use the analysis environment to run the notebooks
   in `results/`. Their names identify the relevant figure and dataset:
   Figures 2–3 and S3–S7 cover model performance; Figures 4–5 cover MOQA
   sequence design and QA search; Figure S8 covers dataset splits and predictor
   performance. Configure notebook paths to the prepared data and stored or
   newly generated outputs. Some notebooks use additional inputs, such as
   `outputs/plm-bvae_metrics.xlsx`; inspect their input cells before execution.
   Run copies if the saved notebook outputs should be preserved.

## Reproducibility notes

The environment specifications are based on the available saved environment
metadata and are not fully validated lock files. Package versions, hardware,
randomness, pretrained model revisions, and D-Wave solver availability can
affect results. The original PARROT installations and some replacement files
are not byte-identical; see the environment READMEs for details. Some analysis
and training paths still require local adjustment, and the peptide Figure 4
sequence-count notebook contains a legacy predictor-output path. Rebuilding
environments and repeating all computations have not been validated as an
end-to-end reproduction.
