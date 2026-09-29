"""Stage-1 candidates: every S1 record -> top-K S2/S3 records of the same country by bi-encoder cosine.

usage: python scripts/gen_candidates.py --split train|test --model models/biencoder_e5s_v1 --k 50 --tag v1
out:   cache/cands_<split>_<tag>.parquet  (r1, r2, rank, score)  r* = row index into cache/<split>_norm.parquet
Countries are taken from the data (open set: France etc. handled the same way).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from er.biencoder import encode, load_model, texts_of  # noqa: E402
from er.embed import gpu_topk  # noqa: E402
from er.io import CACHE, load_norm  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    t0 = time.time()
    df = load_norm(a.split).with_row_index("rid")
    tok, m = load_model(a.model)
    out = []
    for c in sorted(df["country"].unique().to_list()):
        pool = df.filter((pl.col("src") > 1) & (pl.col("country") == c))
        q = df.filter((pl.col("src") == 1) & (pl.col("country") == c))
        t = time.time()
        Ep, Eq = encode(texts_of(pool), tok, m), encode(texts_of(q), tok, m)
        s, i = gpu_topk(Eq, Ep, min(a.k, len(Ep)))
        out.append(pl.DataFrame({"r1": np.repeat(q["rid"].to_numpy(), i.shape[1]).astype(np.uint32),
                                 "r2": pool["rid"].to_numpy()[i].ravel().astype(np.uint32),
                                 "rank": np.tile(np.arange(i.shape[1], dtype=np.int16), len(i)),
                                 "score": s.ravel().astype(np.float16)}))
        print(f"{a.split} {c}: S1 {len(Eq):,} x pool {len(Ep):,} -> {time.time()-t:.0f}s", flush=True)
        del Ep, Eq
    C = pl.concat(out)
    C.write_parquet(CACHE / f"cands_{a.split}_{a.tag}.parquet")
    print(f"saved {C.height:,} candidate pairs in {time.time()-t0:.0f}s", flush=True)
