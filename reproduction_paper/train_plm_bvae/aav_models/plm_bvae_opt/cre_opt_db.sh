source ../../../set_envs/setenv_plm-bvae-py312.sh

python create_optuna_db.py \
       	--db_path "db_optuna/optuna_plm_bvae.db" \
	--study_name "esm_650m"
