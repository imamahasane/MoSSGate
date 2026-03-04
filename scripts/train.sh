#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/training_config.yaml}
OUT=${2:-runs/exp1}

torchrun --standalone --nproc_per_node=2 -m training.train --config ${CONFIG} --out ${OUT}
