#!/bin/bash
# End-to-end: data -> normalise -> split -> bi-encoder -> blocking (candidates) -> matcher -> output/
# Expects ./dataset/{train,test}/*.tsv (copy or symlink the challenge dataset folder here).
set -euo pipefail
cd "$(dirname "$0")/.."   # run from business_entity_resolution/ (dataset/, cache/, models/, output/ live there)
mkdir -p cache models logs output
PY=${PY:-python}

$PY src/scripts/build_norm_cache.py                                   # cache/{train,test}_norm.parquet
$PY src/scripts/make_split.py                                         # cache/split.parquet (fold 0 = validation)
$PY src/scripts/train_biencoder.py --out models/biencoder_e5s_v1 \
    --bs 512 --lr 5e-5 --hard_frac 0.3 --save_every 3000 --stop_step 3000   # fine-tuned multilingual-e5-small
$PY src/scripts/gen_candidates.py --split train --model models/biencoder_e5s_v1 --k 50 --tag v1
$PY src/scripts/gen_candidates.py --split test  --model models/biencoder_e5s_v1 --k 50 --tag v1
$PY src/scripts/train_matcher.py --k 30 --ntrain 300000 --tag m1     # prints val F0.5 + best threshold
$PY src/scripts/predict.py --matcher models/matcher_m1/lgb.txt --k 30 --th 0.675   # output/*.tsv
