source ../../../set_envs/setenv_plm-bvae-py312.sh

# encode test set to binary vectors
python encode_file.py --hparams_path experiment_configs/binary.yml \
	--checkpoint_kind best
