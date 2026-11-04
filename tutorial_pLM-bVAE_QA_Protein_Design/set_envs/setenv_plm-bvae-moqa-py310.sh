#!/bin/bash
export OMP_NUM_THREADS=4
export TF_ENABLE_ONEDNN_OPTS=0

# Define shared directory and paths
ENVDIR="/home/.../local/"
# Initialize conda
export PATH=$ENVDIR/tools/conda/bin:$PATH
source $ENVDIR/tools/conda/etc/profile.d/conda.sh

# Load environment
conda activate plm-bvae-moqa-py310

echo "'plm-bvae-moqa-py310' environment has been initialized."