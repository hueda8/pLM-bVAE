source ../../../set_envs/setenv_parrot-py39.sh

DATADIR=".../data/parrot/peptide"

parrot-train $DATADIR/peptide_random.txt model/network.pt --datatype sequence --classes 1 \
	-lr 0.001511 -nl 2 -hs 58 -b 16 --epochs 10 --stop auto \
       	--split $DATADIR/peptide_indices.txt \
	--encode ../../zscale3_encoding.txt \
       	--include-figs

