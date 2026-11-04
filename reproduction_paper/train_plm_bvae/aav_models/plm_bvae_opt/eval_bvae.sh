source ../../../set_envs/setenv_plm-bvae-py312.sh

# evaluate test set
python evaluate.py --hparams_path experiment_configs/binary.yml \
	--checkpoint_kind best
