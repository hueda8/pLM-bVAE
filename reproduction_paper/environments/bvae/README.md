# bVAE training environment

This directory describes a candidate Python 3.7 environment for the bVAE
training, encoding, and decoding code under [`../../train_bvae/`](../../train_bvae/).
The environment name is `bvae-py37`.

## Create the environment

On a compatible Linux machine with conda installed, run the following commands
from this directory:

```bash
conda env create -f environment.yml
conda activate bvae-py37
```

[`environment.yml`](environment.yml) specifies the Python and TensorFlow
versions found in the saved environment and a focused set of dependencies
used by the bVAE code. It uses pip for NumPy and packages recorded in the
saved Python package metadata. Installing NumPy with pip can replace a conda
package; check the resulting environment before use. This specification is
a **candidate**, not a validated lock file or a complete reproduction of
every package in the saved environment.

## Check the installation

With the environment activated, run:

```bash
python --version
python -m pip check
python -c "import tensorflow as tf, numpy, nltk, multiprocessing_generator, tqdm; print('TensorFlow:', tf.__version__); print('NumPy:', numpy.__version__); print('tf.contrib available:', hasattr(tf, 'contrib'))"
```

The model uses TensorFlow 1.x APIs, including `tf.contrib`; a TensorFlow 2.x
installation is not an equivalent replacement. These checks do not train a
model. Before training, check the dataset and experiment configuration paths
and the imports in the relevant `train_bvae` directory. The provided shell
scripts contain PBS directives and paths specific to the original cluster.

The optional `compare_distance.py` scripts also import Biopython and
`Levenshtein`. Biopython 1.80 is recorded in the saved Python metadata, but a
`Levenshtein` distribution was not found there. These evaluation dependencies
are not included in the candidate training environment; identify and verify
their compatible versions separately before using those scripts.

## What was recorded in the original environment

The saved `bvae_py37/` directory contains Linux conda metadata and Python
package files, but no `bin/` directory. It cannot be used as a complete
executable environment in its present form. Its conda metadata records
Python 3.7.7 and TensorFlow 1.13.1 with an MKL build. It also records PyTorch
1.10.0 with CUDA 10.2; that does not establish that TensorFlow training used
a GPU. Hardware and driver compatibility must be checked on the target
machine.

The saved conda metadata records NumPy 1.19.2 and GPyTorch 1.8.1, whereas
the Python distribution metadata records NumPy 1.21.6 and GPyTorch 1.6.0.
The latter is used for NumPy in the candidate YAML, but the actual runtime
state and the cause of these discrepancies cannot be determined from the
metadata alone. GPyTorch is not included in the focused bVAE specification.

[`observed-python-packages.txt`](observed-python-packages.txt) and
[`observed-conda-packages.txt`](observed-conda-packages.txt) preserve the
two inventories separately. They are informational records, not tested
installation or lock files. Package availability and dependency resolution
may differ on current systems; use the original source and environment
exports, if available, for closer reconstruction.
