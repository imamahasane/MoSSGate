#!/usr/bin/env bash
set -e

CKPT=${1:-runs/exp1/best.pt}
CONFIG=${2:-runs/exp1/config_resolved.yaml}
SPLIT=${3:-test}
OUT_DIR=${4:-runs/exp1/infer_${SPLIT}}

python -m evaluation.run_inference --ckpt ${CKPT} --config ${CONFIG} --split ${SPLIT} --out ${OUT_DIR}
