# Step 0: Prepare the CDRH3 data

This directory prepares Trastuzumab-scFv CDRH3 variant data for pLM-bVAE training and PARROT predictor training. Example prepared files are available in [`../data/`](../data/).

## Procedure

Install Jupyter, pandas, NumPy, seaborn, matplotlib, and Hugging Face `datasets` in the notebook environment. Downloading the source dataset requires network access. Before running the notebooks, update `plm_data` and the input and output paths, and create the required output directories. The example paths use Windows-style separators; adapt them for your platform.

Run the notebooks in this order:

1. [`plm_bvae/cdrh3_data_huggingface.ipynb`](plm_bvae/cdrh3_data_huggingface.ipynb) downloads the YM_0989 dataset, selects HER2 measurements, and prepares the CDRH3 variant sequences and scores.
2. [`plm_bvae/cdrh3_data_random_split.ipynb`](plm_bvae/cdrh3_data_random_split.ipynb) creates training, validation, and test sequence files and residue indices for pLM-bVAE.
3. [`parrot/cdrh3_parrot_dataset.ipynb`](parrot/cdrh3_parrot_dataset.ipynb) prepares sequence-score pairs and split indices for PARROT.

The split notebooks use the same seed and a 0.8:0.1:0.1 ratio. Verify that the sequences in each split match between the pLM-bVAE and PARROT datasets. Use the pLM-bVAE files in [step 1](../step1_train_plm_bvae/) and the PARROT files in [step 2](../step2_train_predictors_parrot/).

[Tutorial overview](../README.md)
