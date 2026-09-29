"""Recall@k of a bi-encoder for blocking: val-fold S1 queries vs the FULL train S2+S3 pool (within country).

usage: python scripts/eval_blocking_dense.py --model <hf id or dir> [--nq 20000] [--tag name]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from er.biencoder import BASE, encode, load_model, texts_of  # noqa: E402
from er.embed import gpu_topk  # noqa: E402
from er.io import CACHE, ROOT, load_gt_pairs, load_norm  # noqa: E402
from er.metrics import blocking_report  # noqa: E402

KS = [1, 5, 10, 20, 50, 100, 200]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--nq", type=int, default=20000, help="val queries per run (half US, half India)")
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()

    t0 = time.time()
    tr = load_norm("train").with_row_index("rid")
    split = pl.read_parquet(CACHE / "split.parquet")
    idmap = tr.select("entity_id", "rid")
    truth = load_gt_pairs().join(idmap, left_on="id1", right_on="entity_id").rename({"rid": "r1"}) \
                           .join(idmap, left_on="id2", right_on="entity_id").rename({"rid": "r2"}).select("r1", "r2")
    val = tr.filter(pl.col("src") == 1).join(split.filter(pl.col("fold") == 0), on="entity_id")
    Q = pl.concat([val.filter(pl.col("country") == c).sample(a.nq // 2, seed=0) for c in ["US", "India"]])
    tok, m = load_model(a.model)

    rows = []
    allc = []
    for c in ["US", "India"]:
        pool = tr.filter((pl.col("src") > 1) & (pl.col("country") == c))
        q = Q.filter(pl.col("country") == c)
        t = time.time()
        Ep = encode(texts_of(pool), tok, m)
        Eq = encode(texts_of(q), tok, m)
        print(f"{c}: encoded pool {len(Ep):,} + q {len(Eq):,} in {time.time()-t:.0f}s", flush=True)
        s, i = gpu_topk(Eq, Ep, max(KS))
        allc.append(pl.DataFrame({"r1": np.repeat(q["rid"].to_numpy(), max(KS)),
                                  "r2": pool["rid"].to_numpy()[i].ravel(),
                                  "rank": np.tile(np.arange(max(KS)), len(i)).astype(np.int16),
                                  "score": s.ravel().astype(np.float32)}))
    C = pl.concat(allc)
    queries = Q.select(pl.col("rid").alias("r1"), "country")
    for k in KS:
        rows.append(blocking_report(C.filter(pl.col("rank") < k), truth, queries, f"{a.tag}@{k}"))
    res = pl.DataFrame(rows)
    with pl.Config(tbl_rows=50, tbl_width_chars=200):
        print(res)
    out = ROOT / "logs" / f"eval_blocking_dense_{a.tag}.csv"
    res.write_csv(out)
    C.write_parquet(CACHE / f"valq_top200_{a.tag}.parquet")
    print(f"saved {out}; total {time.time()-t0:.0f}s")
