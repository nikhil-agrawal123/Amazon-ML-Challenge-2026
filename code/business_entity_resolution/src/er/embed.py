"""Char n-gram TF-IDF -> SVD dense embeddings, and GPU exact top-k search."""
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

HV = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2**20, alternate_sign=False, norm=None)


def _tfidf_chunk(texts, idf):
    X = HV.transform(texts).astype(np.float32)
    X.data = 1.0 + np.log(X.data)  # sublinear tf
    X = X.multiply(idf).tocsr()
    return normalize(X)


class CharSVD:
    """TF-IDF over hashed char 2-4 grams, projected to `dim` with SVD (on GPU), L2-normalised."""

    def __init__(self, dim=256, seed=0):
        self.dim, self.seed = dim, seed

    def fit(self, texts):
        tf = TfidfTransformer(sublinear_tf=True).fit(HV.transform(texts))
        self.idf = tf.idf_.astype(np.float32)
        X = _tfidf_chunk(texts, self.idf).tocoo()
        A = torch.sparse_coo_tensor(np.vstack([X.row, X.col]), X.data, X.shape).cuda()
        torch.manual_seed(self.seed)
        _, _, V = torch.svd_lowrank(A, q=self.dim, niter=4)  # randomized SVD on GPU (~5x faster than sklearn)
        self.W = V.contiguous()  # (2^20, dim)
        return self

    def transform(self, texts, n_jobs=36, chunk=50_000):
        out = np.empty((len(texts), self.dim), np.float16)
        jobs = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
        # sparse tf-idf built in parallel on CPU; projection done on GPU as it streams back
        it = Parallel(n_jobs=n_jobs, return_as="generator")(delayed(_tfidf_chunk)(j, self.idf) for j in jobs)
        pos = 0
        for X in it:
            Xt = torch.sparse_csr_tensor(torch.from_numpy(X.indptr).long(), torch.from_numpy(X.indices).long(),
                                         torch.from_numpy(X.data), size=X.shape).cuda()
            E = torch.nn.functional.normalize(Xt @ self.W, dim=1)
            out[pos:pos + X.shape[0]] = E.half().cpu().numpy()
            pos += X.shape[0]
        return out


@torch.no_grad()
def gpu_topk(Q: np.ndarray, P: np.ndarray, k: int, qbs=1024, pbs=2_000_000):
    """Exact cosine top-k of each row of Q against P (both L2-normalised fp16). Returns (scores, idx)."""
    dev = "cuda"
    Pt = [torch.from_numpy(P[i:i + pbs]).to(dev) for i in range(0, len(P), pbs)]
    out_s, out_i = [], []
    for qs in range(0, len(Q), qbs):
        q = torch.from_numpy(Q[qs:qs + qbs]).to(dev)
        best_s = best_i = None
        off = 0
        for p in Pt:
            s = q @ p.T
            ts, ti = s.topk(min(k, s.shape[1]), dim=1)
            ti = ti + off
            off += p.shape[0]
            if best_s is None:
                best_s, best_i = ts, ti
            else:
                cs, ci = torch.cat([best_s, ts], 1), torch.cat([best_i, ti], 1)
                best_s, j = cs.topk(k, dim=1)
                best_i = ci.gather(1, j)
        out_s.append(best_s.float().cpu().numpy()); out_i.append(best_i.cpu().numpy())
    del Pt
    torch.cuda.empty_cache()
    return np.vstack(out_s), np.vstack(out_i)
