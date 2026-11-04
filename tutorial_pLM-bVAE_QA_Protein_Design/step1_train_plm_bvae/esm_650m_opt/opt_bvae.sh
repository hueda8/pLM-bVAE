source ../../../set_envs/setenv_plm-bvae-moqa-py310.sh

python optuna_optimize.py \
        --hparams_path "experiment_configs/binary.yml" \
        --study_name "esm_650m" \
        --storage "sqlite:///db_optuna/optuna_plm_bvae.db" \
        --n_trials 200 \
        --pruner "none" \
        --sampler "tpe" \
        | tee ./opf_opt.log
