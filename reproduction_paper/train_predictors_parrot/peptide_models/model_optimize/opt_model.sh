source ../../../set_envs/setenv_parrot-py39.sh

DATADIR=".../data/parrot/peptide"

parrot-optimize $DATADIR/peptide_train_dev.txt model_opt/network_optim.py \
        --batch 16 \
        --epochs 100 \
        --max-iter 100 \
        --datatype sequence \
        --classes 1 \
        --set-fractions 0.89 0.1 0.01 \
        --include-figs \
        --save-splits \
	--encode ../../zscale3_encoding.txt \
        --verbose \
        | tee ./opf_opt.log
