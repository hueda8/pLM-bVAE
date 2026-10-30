source ../../../set_envs/setenv_parrot-py39.sh

DATADIR=".../data/parrot"

parrot-train $DATADIR/cdrh3_random.txt model/network.pt --datatype sequence --classes 1 \
	-lr 0.001506 -nl 1 -hs 68 -b 16 --epochs 10 --stop auto \
       	--split $DATADIR/cdrh3_indices.txt \
	--encode ../../zscale3_encoding.txt \
	--include-figs
