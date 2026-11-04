source ../../../set_envs/setenv_plm-bvae-py312.sh

# decode seqs from binary vectors in test set
python decode_file.py --hparams_path experiment_configs/binary.yml \
	--checkpoint_kind best \
	--decode_strategy greedy
