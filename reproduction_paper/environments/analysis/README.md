# Analysis environment

This environment is for the notebooks in `reproduction_paper/results/` and
`reproduction_paper/data_preparation/`. It does not cover model training or
MOQA execution. The direct package versions are taken from the installed
package metadata of the original Python 3.10.11 analysis environment. Jupyter's
`ipykernel` is included so that an existing Jupyter or VS Code installation
can use this environment. `openpyxl` is included for the Excel input used by
the Figure 3 notebook.

## Setup on Windows

From the repository root, with Python 3.10 installed:

```powershell
py -3.10 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r reproduction_paper\environments\analysis\requirements.txt
.\.venv-analysis\Scripts\python.exe -m pip check
```

In VS Code, select `.venv-analysis` as the notebook kernel. Alternatively,
register it with an existing Jupyter installation:

```powershell
.\.venv-analysis\Scripts\python.exe -m ipykernel install --user --name plm-bvae-analysis --display-name "Python (pLM-bVAE analysis)"
```

This requirements file pins direct dependencies, not the full transitive
dependency tree. Resolving and installing it in a fresh environment and
executing the notebooks are still needed to verify the setup. The original
environment could not be started during preparation of this file; versions
were read from its installed distribution metadata instead.

Some notebooks still contain placeholder paths; update those paths before
running them. The Figure 3 notebook expects `outputs/plm-bvae_metrics.xlsx`,
and the CDRH3 Hugging Face preparation notebook downloads its source data.
The necessary input data and outputs must be available separately. For
verification, run copies of the notebooks to preserve their saved outputs.
