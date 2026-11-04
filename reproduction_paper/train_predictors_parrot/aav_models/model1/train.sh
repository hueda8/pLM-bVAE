source ../../../set_envs/setenv_parrot-py39.sh

DATADIR=".../data/parrot/aav"

parrot-train $DATADIR/aav_random.txt model/network.pt --datatype sequence --classes 1 \
	-lr 0.001056 -nl 5 -hs 29 -b 16 --epochs 10 --stop auto \
       	--split $DATADIR/aav_random_indices.txt \
       	--encode ../../zscale3_encoding.txt \
	--include-figs

