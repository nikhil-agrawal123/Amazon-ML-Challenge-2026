# Business Entity Resolution — Neural Ninjas (ML Challenge 2026)

End-to-end pipeline: normalise → fine-tuned bi-encoder → per-country exact top-50 (ScaNN as the scalable drop-in) →
LightGBM filter (~4.3 candidates per S1) → four cross-encoders (e5-small, Laya v1/v2, BGE-reranker-v2-m3) → stacked
LightGBM matchers (+ no-address specialist) → ownership constraint → France map rules + Laya-FR → `output/`.
Method, numbers and all uses of test data: see `Documentation_template.md` at the zip root.

## Two modes (details and measurements: `docs/scalability.md`)
| mode | retrieval | cross-encoders | use for |
|---|---|---|---|
| **`scalable` (default)** | ScaNN per country, 50% of 2,000 partitions, 4-bit codes, CPU (0.3–1.9 ms per query) | India/US: Laya v2 + BGE; France: e5, Laya v1, Laya v2 | large data; any size |
| `exact` | brute-force top-50 per country on GPU (16.6 min on this test set) | all four on every pair | small data; **produced the shipped `output/` (public LB 0.9873)** |

Both give ~4.3 candidates per Source 1 record. ScaNN at 50% retrieves 99.93% (France), 99.98% (India) and 99.99%
(US) of the shipped matches; the lean India/US stack scores 0.98995 vs 0.98993 on validation. After the models are
trained (steps 1-8b below), inference is one command: `bash src/scripts/run_pipeline.sh [--mode exact]`.

## Environment
- Python 3.10.12, `pip install -r requirements.txt` (torch 2.5.1+cu121). Final run: 1× Tesla V100 32 GB (fp16),
  40 CPU cores, 345 GB RAM. BGE-reranker was fine-tuned once on an RTX Blackwell GPU (bf16) with the same recipe as
  `src/scripts/train_hf_ce.py` (see step 6).
- Hugging Face models downloaded on first use: `intfloat/multilingual-e5-small` (MIT, 118M),
  `convaiinnovations/laya-multilingual` (Apache-2.0; encoder weights only, 307M), `BAAI/bge-reranker-v2-m3`
  (Apache-2.0, 568M). LightGBM (MIT). No external data or APIs.

## Layout expected at run time (run every command from this folder)
```
business_entity_resolution/
├── src/er/          library (normalisation, features, competitor features, metrics, cross-encoder, TF-IDF, I/O)
├── src/scripts/     pipeline steps (below)
├── dataset/train/   train_source{1,2,3}.tsv, train_ground_truth.tsv      (copy or symlink the challenge data)
├── dataset/test/    test_source{1,2,3}.tsv
├── cache/  models/  submission/                                          (created by the steps)
```
`export PYTHONPATH=src` before running. Times are for the hardware above.

## Steps
```bash
PY=python
# 1. normalised record caches + S1 folds (fold 0 = validation; bi-encoder folds 1-9, cross-encoders 1-4, matchers 5-9)
$PY src/scripts/build_norm_cache.py && $PY src/scripts/make_split.py                                 # ~3 min
# 2. bi-encoder (InfoNCE, hard negatives), v1 = step 3000
$PY src/scripts/train_biencoder.py --out models/biencoder_e5s_v1 --bs 512 --lr 5e-5 --hard_frac 0.3 --stop_step 3000   # ~27 min
# 3. retrieval: ScaNN top-50 (training pairs) and exact top-50 (final run), per country
$PY src/scripts/gen_candidates_scann.py --split train --model models/biencoder_e5s_v1
$PY src/scripts/gen_candidates_scann.py --split test  --model models/biencoder_e5s_v1
$PY src/scripts/gen_candidates.py --split train --model models/biencoder_e5s_v1 --k 50 --tag exact
$PY src/scripts/gen_candidates.py --split test  --model models/biencoder_e5s_v1 --k 50 --tag exact    # 16.6 min on test
# 4. filter model m2 (pair + competitor features, no cross-encoder) and filtered candidate sets (p2 >= 0.03)
$PY src/scripts/train_matcher.py --k 30 --ntrain 300000 --ctx --cands cands_train_scann --rounds 4000 --tag m2
$PY src/scripts/filter_candidates.py --split train --queries fold5,fold6,fold7,fold8,fold9,fold0 --tag train   # -> filt_tr / filt_va
$PY src/scripts/filter_candidates.py --split train --cands cands_train_exact --queries fold0 --tag va_exact
$PY src/scripts/filter_candidates.py --split test  --cands cands_test_exact --tag test_exact             # 7,407,025 pairs
# 5. cross-encoders (folds 1-4)
$PY src/scripts/train_crossencoder.py --init models/biencoder_e5s_v1 --out models/ce_e5s_v1 --ns1 250000 --k 12
$PY src/scripts/train_crossencoder.py --init <laya-multilingual encoder> --out models/ce_laya_v1 --ns1 100000 --k 12 --bs 128 --lr 2e-5
$PY src/scripts/train_crossencoder.py --init models/ce_laya_v1 --out models/ce_laya_v2 --cands cands_train_scann --ns1 100000 --s1_seed 7 --k 10 --synth_fr 60000 --bs 128 --lr 1e-5
# 6. BGE-reranker-v2-m3: 1 epoch, bs 128, lr 2e-5, max_len 128, 2.81M pairs (folds 1-4 top-k + synthetic French v1)
$PY src/scripts/train_hf_ce.py --init BAAI/bge-reranker-v2-m3 --out models/ce_bge_m3_fp16 --ns1 250000 --k 12 --bs 64 --lr 2e-5
# 7. score every filtered pair with each cross-encoder (--hf for BGE); tags: tr_v2/va_v2/test_v2 (e5), laya_*, laya2_*, bge_*
$PY src/scripts/ce_fill.py --split train --filt filt_tr --model models/ce_laya_v2 --tag laya2_tr        # etc. for each model x {filt_tr, filt_va, filt_va_exact, filt_test_exact}
# 8. matchers: m4 feature list (48 kept of 55), m6 (e5 + Laya v1/v2) for France, m8 (+ BGE) for India/US
bash src/scripts/run_m4.sh      # m4 features.json (--drop of 7 redundant features)
$PY src/scripts/ensemble_matcher.py --seeds 1 --extra laya,laya2 --tag m6
$PY src/scripts/ensemble_matcher.py --seeds 1 --extra laya,laya2,bge --tag m8
# 8b. lean India/US matcher for the scalable mode (no e5 / Laya v1 cross-encoder)
$PY src/scripts/ensemble_matcher.py --seeds 1 --extra laya2,bge --drop_base ce_logit --tag m8lean
# 9. exact-blocking predictions with the saved matchers, ownership fix, no-address specialist C for India/US
$PY src/scripts/predict_saved.py --matcher m6 --extra laya,laya2 --sfx exact --write m6x
$PY src/scripts/ownership_fix.py --pred testpred_m6x --th 0.725 --src submission/output_m6x --out submission/output_m6xo
$PY src/scripts/build_m8xhc.py                         # India/US = m8 + specialist C, France = m6xo -> submission/output_m8xhc
# 10. France map (label-free; logs in Documentation_template.md): category-swap removal + filler-variant adds (+ MAP2)
$PY src/scripts/france_map_variant.py                  # -> submission/output_m8xhcMAP (public LB 0.987)
$PY src/scripts/france_map2.py                         # -> submission/output_m8xhcMAP2
# 10b. Laya-FR (organiser-approved self-training on test-derived French data), run on any CUDA GPU (bf16 on Ampere+):
$PY src/scripts/build_map_labels.py                    # map-labelled French test pairs + generator (france_synth_v4.py --n all first)
#      package data/ for src/scripts/laya_fr/train_score.py (see its README): model = models/ce_laya_v2, train = cache/map_train_fr.parquet,
#      score = all French filtered test pairs; copy out/scores.parquet to laya_fr_kit/out/
$PY src/scripts/build_final_lfd.py                     # -> submission/output_m8xhcLFd  (FINAL, public LB 0.9873)
# 11. final files + validation
cp submission/<final>/matching_results.tsv submission/<final>/candidate_pairs.tsv ../../output/
python3 src/utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Notes
- **Modes**: `bash src/scripts/run_pipeline.sh` runs the scalable default (ScaNN 50% + lean scoring → `submission/output_scalable`);
  `--mode exact` runs steps 3–11 of the exact path. `src/scripts/scann_sweep.py` measures ScaNN coverage/speed, `src/scripts/compare_outputs.py` compares two outputs pair by pair.
- **Indexes**: the scalable run saves per-country ScaNN indexes to `models/scann_index/test_<country>/` (reused when settings match;
  `RESUME=1` reuses existing candidate files). Serve new Source 1 records without rebuilding:
  `python src/scripts/scann_query.py --records new_s1.tsv --index_dir models/scann_index --out new_candidates.tsv`.
- **Use of test data** (organiser-approved: unsupervised statistics, token frequencies, blocking indexes,
  pseudo-labels): retrieval indexes, competitor features and name-frequency counts on test records; the France map
  (`france_synth_v4.py` classifier, `france_map_variant.py`, `france_map2.py`) uses only French test records; India/US
  labels only measure which edit operations are matches. No test labels exist or were used.
- `src/matcher_m4_features.json` is the 48-feature list the final matchers use (output of step 8).
- Determinism: fixed seeds (fold hash 2026, LightGBM seed 0); GPU scoring in fp16 may differ in the 4th decimal.