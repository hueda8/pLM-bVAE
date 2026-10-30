# Environment for pLM-bVAE training and MOQA

This directory describes a Python 3.10 environment for pLM-bVAE training and multi-objective optimization by quantum annealing (MOQA), including prediction with trained PARROT predictors. For predictor training, use the separate [PARROT environment](../parrot/README.md).

The definition is based on the MOQA environment, with Optuna, Biopython, SentencePiece, SciPy, and python-Levenshtein added for pLM-bVAE training and evaluation. It is an **untested candidate**, not a lock file or a complete reproduction of the original experimental environment. The main package versions use the saved Python 3.10 environment as a reference. Ocean SDK and PARROT are pinned to the saved versions, 9.3.0 and 1.7.5, respectively; python-Levenshtein is not pinned.

## Create the environment

On a Linux machine with conda installed, run the following commands from this directory. If an environment with the same name already exists, change `name` in the YAML file first.

```bash
conda env create -f environment.yml
conda activate plm-bvae-moqa-py310
```

The YAML file installs PyTorch with pip, but does not guarantee a particular GPU or CUDA configuration. For GPU use, consult the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to select a build compatible with the machine and driver, and adjust the PyTorch specification before creating the environment if needed. The saved environment was Linux-based; dependency resolution and behavior may differ on other operating systems or CUDA configurations.

## Replace the PARROT predictor for MOQA

After installing PARROT, replace its installed `py_predictor.py` with [`modifed_parrot_files/py_predictor.py`](modifed_parrot_files/py_predictor.py) in this directory. This replacement is required to use the specified amino acid descriptors for prediction in MOQA. With the environment activated, run the following commands **from this directory**. The first command checks that the destination is inside the active environment.

```bash
python -c "import pathlib, shutil, sys, parrot; src=pathlib.Path('modifed_parrot_files/py_predictor.py').resolve(); dst=pathlib.Path(parrot.__file__).resolve().parent / 'py_predictor.py'; assert src.is_file() and dst.is_file(); assert pathlib.Path(sys.prefix).resolve() in dst.parents, 'PARROT is outside the active environment'; shutil.copy2(src, dst); print('Replaced:', dst)"
python -c "import hashlib, pathlib, parrot.py_predictor as p; src=pathlib.Path('modifed_parrot_files/py_predictor.py'); dst=pathlib.Path(p.__file__); assert hashlib.sha256(src.read_bytes()).digest() == hashlib.sha256(dst.read_bytes()).digest(); print('Predictor verified:', dst)"
```

Repeat the replacement if PARROT is reinstalled or upgraded. The supplied file is not identical to the predictor in the saved environment, so applying it does not by itself reproduce the original experimental software state.

## Check the installation

```bash
python --version
python -m pip check
python -c "import torch, numpy, pandas, transformers, peft, optuna, scipy, Bio, Levenshtein, sentencepiece, yaml, tqdm, dimod, botorch, dwave.system, parrot.py_predictor; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Transformers:', transformers.__version__)"
```

These commands check dependencies and imports, but do not verify training or prediction results. On the target machine, use a small dataset to test pLM embeddings, a short pLM-bVAE training run, checkpoint saving and loading, binary encoding, and sequence reconstruction. Then test loading and prediction with a trained PARROT predictor and run MOQA with a small number of candidates. Access to a D-Wave quantum annealer requires additional configuration.

## Before running the tutorial

The training code is under [step1_train_plm_bvae](../../step1_train_plm_bvae/) and the MOQA code is under [step3_run_moqa](../../step3_run_moqa/). Prepare the data, models, predictors, configuration files, and output paths. The MOQA `fmqa` module is local to the run directory, which must be on the Python import path.

The provided shell scripts contain PBS settings, absolute paths specific to the original machine, and `source` commands that activate the original environments. Adapt them to the target machine and activate this shared environment. In particular, check `eval_comp_bvae.sh`, which switches environments during execution. MOQA runs may delete or overwrite existing output files; preserve any results needed before starting a run.

Package versions and CUDA builds can affect results. After validation, save the output of `python -m pip freeze` and `conda env export`, together with the model revisions and experiment configuration.
