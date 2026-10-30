# MOQA environment with pLM-bVAE and PARROT

This directory describes a candidate Python 3.10 environment for the MOQA
samplers combining pLM-bVAE, PARROT predictors, and D-Wave. The environment
name is `moqa-plm-bvae-parrot-py310`. The run scripts and their local `fmqa`
module are under [`../../run_moqa/`](../../run_moqa/).

## Create the environment

On a compatible machine with conda installed, run from this directory:

```bash
conda env create -f environment.yml
conda activate moqa-plm-bvae-parrot-py310
```

[`environment.yml`](environment.yml) specifies Python 3.10.0, the main
packages used by the pLM-bVAE MOQA workflow, and `idptools-parrot` without
the `optimize` extra. The PARROT installation request is equivalent to:

```bash
python -m pip install idptools-parrot
```

The D-Wave installation request is equivalent to:

```bash
python -m pip install dwave-ocean-sdk
```

The saved environment records Ocean SDK 9.3.0, whose dependencies include
`dimod` 0.12.21 and `dwave-system` 1.34.0. They are installed as SDK
dependencies rather than specified separately. Because the SDK is not
pinned, another installation may resolve different versions.

The pinned versions are based on the saved environment's Python package
metadata. This specification is a **candidate**, not a tested lock file or a
complete reproduction of every package in the original environment.

## Replace the PARROT predictor

After installing PARROT and activating the environment, replace its installed
`parrot/py_predictor.py` with the repository's
[`modifed_parrot_files/py_predictor.py`](modifed_parrot_files/py_predictor.py).
This replacement allows PARROT to use the specified amino acid descriptors
when making predictions during MOQA.

Run the following from this directory; it checks that the target is inside
the active environment before overwriting the installed file:

```bash
python -c "import pathlib, shutil, sys, parrot; src=pathlib.Path('modifed_parrot_files/py_predictor.py').resolve(); dst=pathlib.Path(parrot.__file__).resolve().parent / 'py_predictor.py'; assert src.is_file() and dst.is_file(); assert pathlib.Path(sys.prefix).resolve() in dst.parents, 'PARROT is outside the active environment'; shutil.copy2(src, dst); print('Replaced:', dst)"
```

Run the replacement again if PARROT is reinstalled or upgraded. Verify that
the active interpreter loads the intended module and that the file matches:

```bash
python -c "import hashlib, pathlib, parrot.py_predictor as p; src=pathlib.Path('modifed_parrot_files/py_predictor.py'); dst=pathlib.Path(p.__file__); assert hashlib.sha256(src.read_bytes()).digest() == hashlib.sha256(dst.read_bytes()).digest(); print('Predictor verified:', dst)"
```

## Check the installation

```bash
python --version
python -m pip show idptools-parrot dwave-ocean-sdk dimod dwave-system
python -m pip check
python -c "import torch, numpy, dimod, botorch, dwave.system, parrot.py_predictor, yaml, tqdm; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

These checks do not execute MOQA or verify predictions. Each run directory
contains a local `fmqa` module; run the script with that directory on the
Python import path. The shell scripts use cluster-specific PBS directives and
absolute paths. Configure the trained decoder, three predictors, Z-scale
encoding, initial vectors, output paths, and D-Wave access for the target
run. The sampler removes or overwrites specific existing output files, so
preserve any results needed before starting a new run.

## Scope and reproducibility

The saved `moqa_parrot_py310/` directory contains Linux conda metadata and
Python package files, but no `bin/` directory; it is not a complete executable
environment in its present form. Its metadata records Python 3.10.0 and
PARROT 1.7.5. The requested unpinned `pip install idptools-parrot` can install
a different release. The supplied replacement predictor is not byte-identical
to the predictor in the saved environment, so applying it as directed does not
by itself establish that the original software state has been reproduced.

[`observed-python-packages.txt`](observed-python-packages.txt) and
[`observed-conda-packages.txt`](observed-conda-packages.txt) are inventories,
not lock files. The saved environment contains PyTorch and NVIDIA CUDA
packages; the YAML does not guarantee the same PyTorch build or GPU setup.
For another machine, select a compatible build using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) and
verify the resulting environment.

The baseline bVAE decoder scripts use TensorFlow 1.x. This Python 3.10
specification is for the pLM-bVAE MOQA workflow and does not establish a
working baseline bVAE environment. The baseline sampler invokes its decoder
with `python`; using a separate bVAE environment requires adapting and
validating that interpreter selection. The comparison scripts also import
Biopython and `Levenshtein`, which were not found in the saved Python 3.10
metadata; their compatible versions need separate verification.
