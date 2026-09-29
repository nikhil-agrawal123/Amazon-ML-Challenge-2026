# Business Entity Resolution — Team Neural Ninjas

Two-stage pipeline: a fine-tuned multilingual bi-encoder generates candidates (blocking), and a
LightGBM matcher over pair-local string/number features decides the matches.

```
raw TSVs ─► normalise ─► bi-encoder (multilingual-e5-small, fine-tuned) ─► top-30 S2/S3 per S1 (same country)
                                                                             │  = output/candidate_pairs.tsv
                                                                             ▼
                                              46 pair features ─► LightGBM ─► p ≥ 0.675 ─► output/matching_results.tsv
```

## Reproduce

```bash
pip install -r requirements.txt
ln -s /path/to/student_resource/dataset dataset      # needs dataset/{train,test}/*.tsv
export HF_HOME=/some/cache/dir                       # optional: where the base model is downloaded
./src/run_all.sh                                     # or run the 7 steps in it one by one
```

Outputs land in `output/`. Intermediate files go to `cache/`, trained models to `models/`,
and small result tables to `logs/`.

| step | script | what it does | time (V100) |
|---|---|---|---|
| 1 | `src/scripts/build_norm_cache.py` | normalise names/addresses for all 24M records → `cache/*_norm.parquet` | ~2.5 min (36 procs) |
| 2 | `src/scripts/make_split.py` | S1 fold = hash(entity_id) % 10; **fold 0 = validation** | seconds |
| 3 | `src/scripts/train_biencoder.py` | contrastive fine-tune of `intfloat/multilingual-e5-small` on (S1, S2/S3) pairs of folds 1–9; 3000 steps × 512 pairs | ~27 min |
| 4 | `src/scripts/gen_candidates.py` (train, test) | embed all records, exact GPU cosine top-50 within country | ~19 + 17 min |
| 5 | `src/scripts/train_matcher.py` | LightGBM on top-30 candidates of 300k train-fold S1; early stop + threshold sweep on all fold-0 S1 | ~16 min |
| 6 | `src/scripts/predict.py` | score the 52M test candidate pairs, write both TSVs | ~27 min |

Optional: `src/scripts/eval_blocking_dense.py --model models/biencoder_e5s_v1 --tag v1` reports blocking recall@k and
the F0.5 ceiling on validation queries.

## Code layout

```
src/er/io.py          loaders (all TSVs read as strings, tab-separated, no quoting)
src/er/norm.py        unidecode/lowercase/legal-word stripping, consonant skeleton, script detection
src/er/biencoder.py   e5 model loading + length-sorted fp16 encoding
src/er/embed.py       exact GPU top-k (plus a char-n-gram TF-IDF/SVD baseline used during analysis)
src/er/features.py    46 pair-local features (names, addresses, edit-op signature, address numbers)
src/er/metrics.py     macro F0.5 per S1 (singletons count), blocking ceiling
src/scripts/*.py      pipeline steps (see table)
```

## Notes

- Hardware: fp16 everywhere on GPU (V100 has no bf16 tensor cores); the pipeline needs ~60 GB RAM at peak.
- Countries are handled as an open set: blocking is done within each country found in the data (France included),
  and there are no country-specific rules.
- No external data, APIs or geocoding. Only model: `intfloat/multilingual-e5-small` (MIT, 118M parameters).
- Validation (fold 0, 220k S1): blocking F0.5 ceiling @30 ≈ 0.998; end-to-end macro F0.5 = 0.9696.
