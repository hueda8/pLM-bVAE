source ../../../set_envs/setenv_plm-bvae-py312.sh

python embed_ankh.py --mode train \
        --hparams_path ./experiment_configs/binary.yml \
        --amp_dtype fp32

python embed_ankh.py --mode dev \
        --hparams_path ./experiment_configs/binary.yml \
        --amp_dtype fp32

python embed_ankh.py --mode test \
        --hparams_path ./experiment_configs/binary.yml \
        --amp_dtype fp32
