# pLM-bVAE environment

This directory describes a candidate environment for pLM-bVAE training,
protein language model embeddings, hyperparameter optimization, and the
comparison scripts under [`../../train_plm_bvae/`](../../train_plm_bvae/).
The environment name is `plm-bvae-py312`.

## Create the environment

On a compatible machine with conda installed, run the following commands
from this directory:

```bash
conda env create -f environment.yml
conda activate plm-bvae-py312
```

[`environment.yml`](environment.yml) specifies Python 3.12.0 and a focused
set of packages used by the pLM-bVAE code. Most version pins come from the
Python distribution metadata in the saved environment. The comparison code
also imports SciPy and `Levenshtein`, neither of which was found in the saved
environment; `scipy` and `python-Levenshtein` are included without version
pins and require verification. The file is a **candidate**, not a validated
lock file or a complete reproduction of the saved environment.

The saved environment was Linux-based and included NVIDIA CUDA 12.8 runtime
packages. The YAML file requests PyTorch 2.10.0 through pip, but does not
guarantee the same build or CUDA configuration. For a different machine,
select the PyTorch build and compute platform appropriate for it using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/), then
verify the resulting environment. GPU availability also depends on the
machine and driver.

## Check the installation

With the environment activated, run:

```bash
python --version
python -m pip check
python -c "import torch, numpy, transformers, optuna, yaml, tqdm, peft; import pandas, scipy, Bio, Levenshtein, sentencepiece; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Transformers:', transformers.__version__)"
```

These checks do not train a model or verify model outputs. Before running a
training or embedding script, prepare the datasets, experiment configuration,
pLM checkpoints, and output paths. The provided shell scripts include
cluster-specific PBS directives and absolute paths; adapt them to the target
machine. Downloading pretrained models may require network access and
appropriate model access permissions.

## Reproducibility of the saved environment

The saved `plm-bvae_py312/` directory contains Linux conda metadata and Python
package files, but no `bin/` directory; it is not a complete executable
environment in its present form. The conda metadata records Python 3.12.0.
The Python distribution metadata records, among others, PyTorch 2.10.0,
Transformers 5.0.0, NumPy 2.4.2, Optuna 4.7.0, and PEFT 0.18.1.

[`observed-python-packages.txt`](observed-python-packages.txt) and
[`observed-conda-packages.txt`](observed-conda-packages.txt) list the recorded
distributions and conda packages separately. They are informational
inventories, not installation or lock files. Package resolution, platform,
CUDA build, model revisions, and training configuration can affect results.
Use an original environment export and the original model revisions, where
available, for closer reconstruction.

The repository also contains MOQA sampler scripts with additional
dependencies; those are outside the scope of this environment file. The
`optuna_optimize_lora.py` files in the three dataset-specific optimized
model directories currently contain a Python syntax error; they require
correction before that optional workflow can run. The ordinary
`optuna_optimize.py` scripts are separate files.
