source ../../../set_envs/setenv_plm-bvae-py312.sh

python train.py --hparams_path experiment_configs/binary.yml | tee ./opf_train.log
