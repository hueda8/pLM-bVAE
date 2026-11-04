source ../../../set_envs/setenv_plm-bvae-moqa-py310.sh

# encode test set to binary vectors
python encode_file.py --hparams_path experiment_configs/binary.yml \
	--checkpoint_kind best