# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Neural Ninjas  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We treat entity resolution as retrieve-then-verify. A multilingual bi-encoder (`multilingual-e5-small`), fine-tuned
contrastively on the training matches, retrieves the 30 nearest Source 2/3 records for every Source 1 entity within
its country. That keeps 99.5% of true matches (F0.5 ceiling ≈ 0.998). A LightGBM matcher then scores each candidate
pair with 46 pair-local features, including a character-level edit signature and address-number agreement, which catch
the dataset's planted near-duplicate decoys. Validation macro F0.5 is **0.9696**.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** train has 2.2M S1, 5.0M S2 and 5.3M S3 records; test has 1.7M S1 and ~10M S2+S3. Every S2/S3 record
  belongs to at most one S1 (pure 1:N), and there are 3.46 matches per S1 on average (1.7 in S2, 1.8 in S3).
  5.6% of S1 are singletons, and ~26% of S2/S3 records are unmatched distractors.
- **Country** agrees on 100% of true pairs, so blocking within country loses nothing. France appears only in test.
- **Scripts:** in India, 23% of S2 names and 13% of S3 names are non-Latin (Devanagari plus 8 other scripts), e.g.
  "रियल अरिहंत कंस्ट्रक्शंस" ↔ "Real Arihant Constructions". There are no PIN codes. States appear as full names,
  2-letter codes or in native script.
- **Name noise:** legal suffix changes, word reordering, bracketed words, junk prefixes ("-- ", "#"), OCR-style
  digit↔letter swaps ("K0lkata"), domain-style names ("davismann.com"), acronyms ("ET" ↔ "Easy Tools"), and fully
  replaced brand names at an identical address.
- **Address noise:** abbreviations, reordered components, missing street numbers, empty or `<NULL>` / `N/A`
  addresses (~3%), and uppercase formatting in S2.
- **Decoys:** unmatched records that copy an S1 with one letter doubled ("Ferrora" → "ferrrora") or with a shifted
  house or unit number, at the same address. Name collisions are also frequent (dozens of different "Ujjaraj …"
  companies). Among candidates with an identical name and near-identical address, P(match) = 1.000 when all address
  numbers agree and 0.82 when they don't.
- **Train → test shift:** name and address length distributions are identical (KS D ≤ 0.034). Test has ~23% more
  S2/S3 records per S1 and a different country mix. We therefore avoid density-dependent features (candidate count,
  rank, gap-to-best) and match caps.

### 2.2 Solution Strategy
**Approach Type:** Blocking (fine-tuned dense retrieval) + gradient-boosted pair classifier  
**Core Innovation:** A contrastively fine-tuned multilingual bi-encoder closes the cross-script gap in blocking
(India F0.5 ceiling @10 rises from 0.972 to 0.997). Decoy-aware pair features — an edit-operation signature on the
un-normalised names plus address-number agreement — keep F0.5 precision high.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** dense retrieval. Each record is embedded as `query: {name} | {address}` with fine-tuned
  `multilingual-e5-small` (mean pooling, 384-d, fp16), and we take the exact cosine top-30 of S2+S3 for each S1
  within the same country. We ran an exhaustive GPU matrix product rather than approximate search.
- **Fine-tuning:** symmetric InfoNCE (scale 20) with in-batch negatives. Batches are single-country, 30% of batches
  are grouped by name prefix to create harder negatives, and other positives of the same S1 are masked out.
  3000 steps × 512 pairs; S1 folds 1–9 only.
- **Candidate pairs generated:** 52.0M for test (30 per S1). Final `candidate_pairs.tsv` = exactly the list the
  matcher scores.
- **How we ensured true matches were not lost** (validation, 20k fold-0 queries vs the full train pool):

| method | pair recall | F0.5 ceiling | cands/S1 |
|---|---|---|---|
| exact keys (best: house no. + first name token) | 0.55 | – | 34 |
| rare-token overlap (address, 3 rarest, df ≤ 10k) | 0.899 | 0.958 | 4,733 |
| char 2–4-gram TF-IDF → SVD, name+address @50 | 0.769 | 0.895 | 50 |
| e5-small, not fine-tuned @10 | 0.941 | 0.981 | 10 |
| **e5-small fine-tuned @10** | **0.990** | **0.997** | 10 |
| **e5-small fine-tuned @30 (used)** | ≈0.996 | ≈0.998 | 30 |

---

## 4. Matching Model

**Features used (46, all pair-local):**
- Name features: fuzzy ratio / partial / token-set / token-sort on the normalised name; token-set, ratio and
  Jaro-Winkler on the core name (legal words removed); consonant-skeleton similarity (for transliterated Indic names);
  core-token Jaccard and containment; first-token equality; acronym match; domain-name match; length ratio;
  target script.
- Address features: ratio / partial / token-set / token-sort; token Jaccard and containment; Jaccard and containment
  of number sets; house-number equality; last-component (state/city) equality; empty-address flag; length ratio.
- Decoy features: Levenshtein edit-op signature on the un-collapsed core name (letter-doubling insert or delete,
  digit↔letter substitution, other substitution / insert / delete); address numbers equal, subset, first-number
  equality and relative difference.
- Other: bi-encoder cosine similarity. No candidate-count, rank or ID features, because those would not transfer to
  the denser test set.

**Model type:** LightGBM binary classifier (127 leaves, learning rate 0.05, 3000 rounds, early stopping on
validation). Trained on 9.0M candidate pairs from 300k train-fold S1.  
**Threshold selection method:** swept p ∈ [0.30, 0.95] for macro F0.5 on all 220k fold-0 S1 (singletons included).
The best is p = 0.675, and the curve is flat (0.969–0.970) over 0.625–0.75.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation fold 0 (220k S1):** 0.9696 (India 0.9681, US 0.9705). Train-fold F0.5 is 0.9803.
- **Common false positives (wrong merges):** decoy copies of an S1 at the same address with a doubled letter or
  shifted unit number; identical names at a nearby address; different companies in the same building.
- **Common false negatives (missed matches):** fully replaced brand names ("Nexpyracalo") with a perturbed house
  number; domain-style names ("deliennis.com"); heavily truncated S2/S3 addresses combined with native-script names.

---

## 6. Conclusion
Fine-tuning a small multilingual encoder essentially solves blocking for this data (F0.5 ceiling ≈ 0.998 at
30 candidates) and handles native-script names without transliteration rules. Most of the remaining error is
precision against planted decoys, where character-level edit signatures and address-number agreement matter more
than generic string similarity.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `src/run_all.sh` runs the whole pipeline. Steps: `build_norm_cache` → `make_split`
→ `train_biencoder` → `gen_candidates` (train, test) → `train_matcher` → `predict`. Library code is in `src/er/`.
See its README for timings and the layout.

### B. Additional Results
Bi-encoder recall@k on validation (fine-tuned): @5 0.914, @10 0.990, @20 0.995, @50 0.997, @200 0.999.
Feature gain ranking: bi-encoder cosine ≫ address token-set > address numbers equal > core-name Jaro-Winkler >
first-number relative difference > number Jaccard.
