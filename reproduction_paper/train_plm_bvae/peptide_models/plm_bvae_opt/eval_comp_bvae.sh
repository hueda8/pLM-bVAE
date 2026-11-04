source ../../../set_envs/setenv_plm-bvae-py312.sh

# evaluate test set
python evaluate.py --hparams_path experiment_configs/binary.yml \
	--checkpoint_kind best

# encode test set to binary vectors
python encode_file.py --hparams_path experiment_configs/binary.yml \
	--checkpoint_kind best

# decode seqs from binary vectors in test set
python decode_file.py --hparams_path experiment_configs/binary.yml \
	--checkpoint_kind best

conda deactivate

# evaluation of reconstruction and compression performances of Previous and Modified model
source ../../../set_envs/setenv_plm-bvae-py312.sh

DATADIR=".../data/peptide/bvae"
MODELDIR=".../peptide_models/esm_650m"

python compare_distance.py --test_seqs $DATADIR/test.txt \
        --decoded_seqs1 $MODELDIR/model_output/binary/decoded.txt --decoded_seqs2 ./model_output/binary/decoded.txt \
        --binary_seqs1 $MODELDIR/model_output/binary/vectors.txt --binary_seqs2 ./model_output/binary/vectors.txt \
        --output ./model_output/binary/compare_results.csv \
	--indices_dir ../embeddings_esm_650m/test/indices
