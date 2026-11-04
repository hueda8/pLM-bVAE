source ../../../set_envs/setenv_plm-bvae-py312.sh
python list_optuna_studies.py \
       	--storage-url "sqlite:///db_optuna/optuna_plm_bvae.db" \
	--delete-study "esm_650m" 
