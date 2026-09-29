"""Stage-2 matcher: LightGBM on pair-local features over v1 bi-encoder top-K candidates.

- train rows: candidates of --ntrain S1 from folds 1..9; val rows: candidates of ALL fold-0 S1
- early stopping on val; threshold swept for macro F0.5 on val (singletons included)
- saves model + val predictions (r1, r2, p, y) for post-processing / stacking

usage: python scripts/train_matcher.py --k 30 --ntrain 300000 --tag m1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from er.features import FIELDS, NAMES, pair_features  # noqa: E402
from er.io import CACHE, ROOT, load_gt_pairs, load_norm  # noqa: E402
from er.metrics import f05_per_entity  # noqa: E402


def build(C, recs, truth):
    C = C.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["r1", "r2"], how="left").fill_null(0)
    A = C.join(recs, left_on="r1", right_on="rid", how="left").select(FIELDS).rows()
    B = C.join(recs, left_on="r2", right_on="rid", how="left").select(FIELDS).rows()
    X = np.hstack([pair_features(A, B, procs=32), C["score"].cast(pl.Float32).to_numpy()[:, None]])
    return C, X


def sweep(ev, truth_q, ths):
    """ev: (r1, p, y) for all candidates of the query set; truth_q: (r1, k) incl. singletons (k=0)."""
    res = []
    for th in ths:
        per = ev.group_by("r1").agg((pl.col("p") >= th).sum().alias("npred"),
                                    ((pl.col("p") >= th) & (pl.col("y") == 1)).sum().alias("found"))
        per = truth_q.join(per, on="r1", how="left").fill_null(0)
        f = per.with_columns(f05_per_entity(pl.col("k"), pl.col("found"), pl.col("npred")).fill_nan(0).alias("f"))
        res.append({"th": round(float(th), 3), "F05": f["f"].mean(),
                    **{f"F05_{c}": f.filter(pl.col("country") == c)["f"].mean() for c in sorted(f["country"].unique())}})
    return pl.DataFrame(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--ntrain", type=int, default=300_000)
    ap.add_argument("--cands", default="cands_train_v1")
    ap.add_argument("--tag", default="m1")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_norm("train").with_row_index("rid")
    split = pl.read_parquet(CACHE / "split.parquet")
    idmap = tr.select("entity_id", "rid")
    truth = load_gt_pairs().join(idmap, left_on="id1", right_on="entity_id").rename({"rid": "r1"}) \
                           .join(idmap, left_on="id2", right_on="entity_id").rename({"rid": "r2"}).select("r1", "r2")
    truth = truth.with_columns(pl.col("r1").cast(pl.UInt32), pl.col("r2").cast(pl.UInt32))
    recs = tr.select(pl.col("rid").cast(pl.UInt32), *FIELDS)
    s1 = tr.filter(pl.col("src") == 1).select(pl.col("rid").cast(pl.UInt32).alias("r1"), "entity_id", "country").join(split, on="entity_id")
    C = pl.read_parquet(CACHE / f"{a.cands}.parquet").filter(pl.col("rank") < a.k)

    trq = s1.filter(pl.col("fold") != 0).sample(a.ntrain, seed=0).select("r1")
    vaq = s1.filter(pl.col("fold") == 0).select("r1", "country")
    Ct, Xt = build(C.join(trq, on="r1"), recs, truth)
    print(f"train rows {len(Xt):,} pos {Ct['y'].sum():,} | {time.time()-t0:.0f}s", flush=True)
    Cv, Xv = build(C.join(vaq.select("r1"), on="r1"), recs, truth)
    print(f"val rows {len(Xv):,} pos {Cv['y'].sum():,} | {time.time()-t0:.0f}s", flush=True)

    names = NAMES + ["dense_score"]
    prm = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
           "verbose": -1, "num_threads": 32}
    dt = lgb.Dataset(Xt, Ct["y"].to_numpy(), feature_name=names)
    dv = lgb.Dataset(Xv, Cv["y"].to_numpy(), reference=dt)
    mdl = lgb.train(prm, dt, 3000, valid_sets=[dt, dv], valid_names=["train", "val"],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    pv = mdl.predict(Xv, num_iteration=mdl.best_iteration)
    pt = mdl.predict(Xt, num_iteration=mdl.best_iteration)

    kq = vaq.join(truth.group_by("r1").len().rename({"len": "k"}), on="r1", how="left").fill_null(0)
    ev = Cv.select("r1", "r2", "y").with_columns(pl.Series("p", pv))
    res = sweep(ev, kq, np.arange(0.3, 0.97, 0.025))
    best = res.sort("F05", descending=True).row(0, named=True)
    with pl.Config(tbl_rows=40):
        print(res)
    ktr = trq.join(s1.select("r1", "country"), on="r1").join(truth.group_by("r1").len().rename({"len": "k"}), on="r1", how="left").fill_null(0)
    f_tr = sweep(Ct.select("r1", "r2", "y").with_columns(pl.Series("p", pt)), ktr, [best["th"]])["F05"][0]
    print(f"BEST val F0.5 {best['F05']:.4f} @th {best['th']} | train F0.5 @same th {f_tr:.4f} | best_iter {mdl.best_iteration}")
    imp = sorted(zip(names, mdl.feature_importance("gain")), key=lambda x: -x[1])
    print("gain:", [(n, int(g)) for n, g in imp[:20]])
    out = ROOT / "models" / f"matcher_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    mdl.save_model(str(out / "lgb.txt"), num_iteration=mdl.best_iteration)
    ev.write_parquet(CACHE / f"valpred_{a.tag}.parquet")
    res.write_csv(ROOT / "logs" / f"matcher_{a.tag}_threshold_sweep.csv")
    print(f"done {time.time()-t0:.0f}s")
