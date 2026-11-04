# PARROT environment for predictor training

This directory describes how to set up a Python 3.9 environment with PARROT
and its optional hyperparameter optimization dependencies. The predictor
training scripts are under [`../../train_predictors_parrot/`](../../train_predictors_parrot/).

## Create the environment

Install conda and run the following commands in a terminal where `conda` is
available. These commands follow the original setup procedure:

```bash
conda create --name parrot-py39 python=3.9
conda activate parrot-py39
python -m pip install "idptools-parrot[optimize]"
```

The quotes around the package name also make the command suitable for shells
that interpret square brackets. The `optimize` extra is required for
`parrot-optimize`. See the [PARROT installation instructions](https://github.com/idptools/parrot/blob/master/docs/getting_started.rst).

Alternatively, from this directory, create the environment from
[`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate parrot-py39
```

The YAML file selects the `conda-forge` channel and uses the same Python
minor version and pip package request. Use a different environment name if
`parrot-py39` already exists. The name differs from the original
saved environment so that the two can be kept separate.

## Replace the PARROT training files

After installing PARROT and activating the environment, replace the installed
`parrot/scripts/parrot_train.py` and `parrot/tools/validate_args.py` with the
corresponding files in [`modifed_parrot_files/`](modifed_parrot_files/). These
modifications allow predictor training to use the same training and validation
dataset splits as bVAE and pLM-bVAE. Supply the matching datasets and splits
when running the training scripts.

Run the following from this directory. It checks that both source and target
files exist and that the installed PARROT package is inside the active
environment before replacing the files:

```bash
python -c "import pathlib, shutil, sys, parrot; root=pathlib.Path(parrot.__file__).resolve().parent; pairs=[(pathlib.Path('modifed_parrot_files') / name, root / subdir / name) for subdir, name in [('scripts', 'parrot_train.py'), ('tools', 'validate_args.py')]]; assert pathlib.Path(sys.prefix).resolve() in root.parents, 'PARROT is outside the active environment'; assert all(src.is_file() and dst.is_file() for src, dst in pairs), 'Source or installed file is missing'; [(shutil.copy2(src, dst), print('Replaced:', dst)) for src, dst in pairs]"
```

Verify both replacements with:

```bash
python -c "import hashlib, pathlib, parrot; root=pathlib.Path(parrot.__file__).resolve().parent; pairs=[(pathlib.Path('modifed_parrot_files') / name, root / subdir / name) for subdir, name in [('scripts', 'parrot_train.py'), ('tools', 'validate_args.py')]]; assert all(hashlib.sha256(src.read_bytes()).digest() == hashlib.sha256(dst.read_bytes()).digest() for src, dst in pairs), 'Replacement files do not match'; print('Both PARROT files verified')"
```

Repeat the replacement after reinstalling or upgrading PARROT. Because the
installed PARROT version is not pinned, compatibility of these replacement
files with a newly resolved version must be checked.

## Check the installation

With the environment activated, check the interpreter, dependency metadata,
and command-line entry points:

```bash
python --version
python -m pip show idptools-parrot GPy GPyOpt torch
python -m pip check
parrot-train --help
parrot-optimize --help
parrot-predict --help
```

These checks verify the installation and command availability; they do not
reproduce model training. For actual training, provide the appropriate
datasets, splits, encoding file, and output paths. The scripts in
`train_predictors_parrot/` contain cluster-specific PBS directives and
absolute paths that must be adapted to the target machine.

## Reproducibility of the original environment

The directory `parrot_train_final_paper_py39/` contains a saved Linux conda
environment. Its metadata records Python **3.9.0**, and the installed
distribution metadata identifies PARROT as
`1.0.0+21.gc6c17ce.dirty`, installed from a local file source. The `dirty`
version indicates a locally modified source state; the version alone does not
identify all changes. The recorded Python package versions are listed in
[`observed-packages.txt`](observed-packages.txt) for reference. This list is an
inventory, not a tested lock file or an installation command.

Installing `idptools-parrot[optimize]` without version constraints resolves
packages available at installation time. Consequently, the commands above
reproduce the documented setup procedure, but do not guarantee the same
PARROT code, dependencies, GPU configuration, or training results as the
saved environment. For exact reconstruction, obtain the original modified
PARROT source and an export of the original environment, and verify them on a
compatible platform. The saved environment is Linux-specific and should not
be assumed to work by copying it to another machine.

The two replacement files provided in this directory are required for the
dataset splits described above. They do not by themselves establish that the
original installed PARROT source has been reconstructed exactly. The repository
also contains `train_predictors_parrot/modifed_parrot_files/`; its three files
are not byte-identical to the corresponding files in the saved Python 3.9
environment. Some
training and optimization scripts refer to a separate `py312` environment,
so this Python 3.9 setup should not be assumed to reproduce those runs.
