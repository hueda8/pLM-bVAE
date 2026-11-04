source ../../../set_envs/setenv_bvae-py37.sh

# evaluate test set
python evaluate.py experiment_configs/binary.json

# encode test set to binary vectors
python encode_file.py experiment_configs/binary.json

# decode seqs from binary vectors in test set
python decode_file.py experiment_configs/binary.json

conda deactivate

# evaluation of reconstruction and compression performances of Previous and Modified model
source ../../../set_envs/setenv_bvae-py37.sh

DATADIR=".../data/aav/bvae"
MODELDIR=".../aav_models/bvae"

python compare_distance.py --test_seqs $DATADIR/test.txt \
        --decoded_seqs1 $MODELDIR/model_output/binary/decoded.txt --decoded_seqs2 ./model_output/binary/decoded.txt \
        --binary_seqs1 $MODELDIR/model_output/binary/vectors.txt --binary_seqs2 ./model_output/binary/vectors.txt \
        --output ./model_output/binary/compare_results.csv
