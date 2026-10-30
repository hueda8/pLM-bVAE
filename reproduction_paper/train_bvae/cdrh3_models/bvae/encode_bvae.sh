source ../../../set_envs/setenv_bvae-py37.sh

# encode test set to binary vectors
python encode_file.py experiment_configs/binary.json
