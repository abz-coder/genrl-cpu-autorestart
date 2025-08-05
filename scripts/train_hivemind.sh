#!/bin/bash

source ~/.profile
ROOT=$PWD

python "$ROOT/src/genrl/runner/hivemind_launcher.py" \
    --config-path "$ROOT/recipes/$1" \
    --config-name $2
