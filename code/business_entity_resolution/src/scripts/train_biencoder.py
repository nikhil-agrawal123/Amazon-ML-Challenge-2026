"""Contrastive fine-tuning of a multilingual bi-encoder on (S1, S2/S3) GT pairs, for blocking.

- trains only on S1 entities in folds 1..9 (fold 0 = val, see make_split.py)
- in-batch negatives, symmetric InfoNCE, scale 20; batches are single-country (harder negatives)
- other positives of the same S1 inside a batch are masked out (they are not negatives)
- --hard_frac: fraction of batches built from pairs sorted by a name/address key -> in-batch hard negatives
- fp32 master weights + fp16 autocast (V100)

usage: python scripts/train_biencoder.py --out models/<name> [--bs 512 --lr 5e-5 --max_pairs 0 --epochs 1]
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from er.biencoder import BASE, MAX_LEN, load_model, make_text, mean_pool  # noqa: E402
from er.io import CACHE, ROOT, load_gt_pairs, load_norm  # noqa: E402


class PairBatches(Dataset):
    def __init__(self, a, b, grp, batches, tok, hn=None, hn_owner=None):
        self.a, self.b, self.grp, self.batches, self.tok = a, b, grp, batches, tok
        self.hn, self.hn_owner = hn, hn_owner  # hn[i]: list of hard-negative texts for pair i; owner: their S1 group (-1 = none)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = self.batches[i]
        ta = self.tok([self.a[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        tb = self.tok([self.b[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        g = torch.from_numpy(self.grp[idx])
        if self.hn is None:
            return ta["input_ids"], ta["attention_mask"], tb["input_ids"], tb["attention_mask"], g, None, None, None
        rng = np.random.default_rng()
        pick = [rng.integers(len(self.hn[j])) if len(self.hn[j]) else -1 for j in idx]
        txt = [self.hn[j][k] if k >= 0 else self.b[j] for j, k in zip(idx, pick)]
        own = [self.hn_owner[j][k] if k >= 0 else self.grp[j] for j, k in zip(idx, pick)]  # no hn -> own positive (masked)
        th = self.tok(txt, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return (ta["input_ids"], ta["attention_mask"], tb["input_ids"], tb["attention_mask"], g,
                th["input_ids"], th["attention_mask"], torch.tensor(own, dtype=torch.long))


def make_batches(country, key, bs, hard_frac, rng):
    """Single-country batches. A `hard_frac` share are contiguous runs after sorting by `key` (similar records together)."""
    batches = []
    for c in np.unique(country):
        ix = np.where(country == c)[0]
        rng.shuffle(ix)
        n_hard = int(len(ix) * hard_frac)
        hard, rand = ix[:n_hard], ix[n_hard:]
        hard = hard[np.argsort(key[hard], kind="stable")]
        for part in (hard, rand):
            batches += [part[i:i + bs] for i in range(0, len(part) - bs + 1, bs)]
    rng.shuffle(batches)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=0)
    ap.add_argument("--hard_frac", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=3000)
    ap.add_argument("--stop_step", type=int, default=0, help="stop early (LR schedule still sized for the full run)")
    ap.add_argument("--hardneg", default="", help="cache/hardneg_<tag>.parquet from mine_hardneg.py")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    print(vars(a), flush=True)

    tr = load_norm("train")
    split = pl.read_parquet(CACHE / "split.parquet")
    rec = tr.select("entity_id", "business_name", "business_address", "country", "name_core")
    P = load_gt_pairs().join(split.filter(pl.col("fold") != 0), left_on="id1", right_on="entity_id") \
        .join(rec, left_on="id1", right_on="entity_id") \
        .join(rec.select("entity_id", "business_name", "business_address"), left_on="id2", right_on="entity_id", suffix="_r")
    hn = hn_owner = None
    if a.hardneg:
        rid = tr.select("entity_id").with_row_index("rid")
        H = pl.read_parquet(a.hardneg).join(rid, left_on="r1", right_on="rid").rename({"entity_id": "id1h"}) \
              .join(rid, left_on="r2", right_on="rid").rename({"entity_id": "idh"})
        own = load_gt_pairs().select(pl.col("id2").alias("idh"), pl.col("id1").alias("owner"))
        H = H.join(own, on="idh", how="left").join(rec.select("entity_id", "business_name", "business_address"),
                                                   left_on="idh", right_on="entity_id")
        H = H.with_columns(pl.concat_str(pl.lit("query: "), "business_name", pl.lit(" | "), "business_address").alias("t"))
        Hg = H.group_by("id1h").agg("t", "owner")
        P = P.join(Hg, left_on="id1", right_on="id1h", how="inner")
    if a.max_pairs:
        P = P.sample(min(a.max_pairs, P.height), seed=0)
    print("train pairs:", P.height, flush=True)
    A = [make_text(n, s) for n, s in zip(P["business_name"].to_list(), P["business_address"].to_list())]
    B = [make_text(n, s) for n, s in zip(P["business_name_r"].to_list(), P["business_address_r"].to_list())]
    cats = P["id1"].unique().to_list()
    code = {k: i for i, k in enumerate(cats)}
    grp = np.array([code[k] for k in P["id1"].to_list()], dtype=np.int64)
    if a.hardneg:
        hn = P["t"].to_list()
        hn_owner = [[code.get(o, -1) if o is not None else -1 for o in os_] for os_ in P["owner"].to_list()]
    country = P["country"].to_numpy()
    key = P.select(pl.col("name_core").str.slice(0, 6))["name_core"].to_numpy()
    del tr, P

    tok, model = load_model(a.init, train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(0)
    steps_per_epoch = len(A) // a.bs
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * max(0.0, 1 - s / total))
    print(f"steps: {total}", flush=True)

    step, t0, run_loss = 0, time.time(), 0.0
    for ep in range(a.epochs):
        batches = make_batches(country, key, a.bs, a.hard_frac, rng)
        dl = DataLoader(PairBatches(A, B, grp, batches, tok, hn, hn_owner), batch_size=None, shuffle=False, num_workers=12, prefetch_factor=8)
        for ia, ma, ib, mb, g, ih, mh, gh in dl:
            ia, ma, ib, mb, g = (x.cuda(non_blocking=True) for x in (ia, ma, ib, mb, g))
            with torch.autocast("cuda", dtype=torch.float16):
                ea = mean_pool(model(input_ids=ia, attention_mask=ma).last_hidden_state, ma)
                eb = mean_pool(model(input_ids=ib, attention_mask=mb).last_hidden_state, mb)
                if ih is not None:
                    ih, mh, gh = ih.cuda(non_blocking=True), mh.cuda(non_blocking=True), gh.cuda(non_blocking=True)
                    eh = mean_pool(model(input_ids=ih, attention_mask=mh).last_hidden_state, mh)
            ea, eb = F.normalize(ea.float(), dim=-1), F.normalize(eb.float(), dim=-1)
            logits = a.scale * ea @ eb.T
            same = (g[:, None] == g[None, :]) & ~torch.eye(len(g), dtype=torch.bool, device=g.device)
            logits = logits.masked_fill(same, -1e4)
            lab = torch.arange(len(g), device=g.device)
            if ih is not None:
                # q -> [in-batch positives ; hard negatives]; mask hard negs that belong to the row's own S1
                lh = (a.scale * ea @ F.normalize(eh.float(), dim=-1).T).masked_fill(g[:, None] == gh[None, :], -1e4)
                loss = (F.cross_entropy(torch.cat([logits, lh], 1), lab) + F.cross_entropy(logits.T, lab)) / 2
            else:
                loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += loss.item()
            if step % 100 == 0:
                el = time.time() - t0
                acc = (logits.argmax(1) == lab).float().mean().item()
                print(f"ep {ep} step {step}/{total} loss {run_loss/100:.4f} inbatch_acc {acc:.3f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{step*a.bs/el:.0f} pairs/s eta {(total-step)*el/step/60:.1f}m", flush=True)
                run_loss = 0.0
                if not math.isfinite(loss.item()):
                    raise SystemExit("non-finite loss")
            if step % a.save_every == 0:
                model.save_pretrained(out); tok.save_pretrained(out)
            if a.stop_step and step >= a.stop_step:
                break
        if a.stop_step and step >= a.stop_step:
            break
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"done {step} steps in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
