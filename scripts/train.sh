#!/bin/bash

# source ~/.profile
ROOT=$PWD

torchrun \
    --nnodes=$1 \
    --nproc-per-node=$2 \
    --rdzv-id=genrl.swarm.run \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$ROOT/src/genrl/runner/launcher.py" \
    --config-path "$ROOT/recipes/$3" \
    --config-name $4
