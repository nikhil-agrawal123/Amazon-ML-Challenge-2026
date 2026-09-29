"""Pairwise features for the matcher. Pair-local only: no block size / candidate count / ID features
(test has ~23% more S2/S3 records per S1 than train, so density-dependent features would not transfer).

Input rows are dicts/tuples of the normalised fields from cache/*_norm.parquet for both records.
"""
import re
from multiprocessing import Pool

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode

from er.norm import SCRIPTS

FIELDS = ["name_n", "name_core", "name_skel", "addr_n", "hnum", "addr_last", "name_script", "business_name"]
_RAWTOK = re.compile(r"[^a-z0-9 ]+")
# legal words for the raw (uncollapsed) core; mirrors norm.LEGAL but spelled un-collapsed
_RAW_LEGAL = {"pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
              "llp", "lp", "plc", "pc", "pllc", "public", "the", "and", "of", "group", "groupe", "sarl", "sas", "sasu",
              "eurl", "sa", "sci", "dba", "india", "usa", "france", "com", "www"}
_NUM = re.compile(r"\d+")
_DOMAIN = re.compile(r"(?i)\.(com|in|net|org|co|fr)\b")
_SCRIPT_ID = {"latin": 0, **{n: i + 1 for i, (n, _, _) in enumerate(SCRIPTS)}, "other": len(SCRIPTS) + 1}

NAMES = [
    # name
    "n_ratio", "n_partial", "n_tset", "n_tsort", "core_tset", "core_ratio", "core_jw", "skel_tset", "skel_ratio",
    "core_jacc", "core_contain_min", "core_first_eq", "acronym", "domain_match", "b_is_domain",
    "core_len_ratio", "core_ntok_a", "core_ntok_b", "b_script", "b_nonlatin",
    # address
    "a_ratio", "a_partial", "a_tset", "a_tsort", "a_jacc", "a_contain_b", "num_jacc", "num_b_in_a", "hnum_eq",
    "hnum_both", "last_eq", "b_addr_empty", "a_len_ratio",
    # raw-name edit signature (repeat letters NOT collapsed: letter-doubling is a decoy operator in this data)
    "raw_ed", "raw_exact", "op_ins_dup", "op_del_dedup", "op_sub_digit", "op_sub_other", "op_ins_other", "op_del_other",
    # address numbers (house/unit shifts are a decoy operator)
    "nums_eq", "nums_b_sub_a", "num_first_eq", "num_first_reldiff",
]


def _raw_core(raw):
    return " ".join(t for t in _RAWTOK.sub(" ", unidecode(raw).lower()).split() if t not in _RAW_LEGAL)


def _edit_sig(a, b):
    """counts of edit-op classes a->b; nan if names are far apart (op mix is only meaningful for near-identical names)."""
    d = Levenshtein.distance(a, b)
    if d > 3 or not a or not b:
        return [d, float(a == b)] + [np.nan] * 6
    c = [0] * 6  # ins_dup, del_dedup, sub_digit, sub_other, ins_other, del_other
    for op in Levenshtein.editops(a, b):
        if op.tag == "insert":
            ch = b[op.dest_pos]
            dup = (op.src_pos < len(a) and a[op.src_pos] == ch) or (op.src_pos > 0 and a[op.src_pos - 1] == ch)
            c[0 if dup else 4] += 1
        elif op.tag == "delete":
            ch = a[op.src_pos]
            dup = (op.src_pos + 1 < len(a) and a[op.src_pos + 1] == ch) or (op.src_pos > 0 and a[op.src_pos - 1] == ch)
            c[1 if dup else 5] += 1
        else:
            c[2 if a[op.src_pos].isdigit() != b[op.dest_pos].isdigit() else 3] += 1
    return [d, float(a == b)] + c


def _jacc(x, y):
    if not x or not y:
        return np.nan
    return len(x & y) / len(x | y)


def _contain(x, y):
    """share of the smaller set contained in the other"""
    if not x or not y:
        return np.nan
    return len(x & y) / min(len(x), len(y))


def _acronym(core_a, core_b):
    ta, tb = core_a.split(), core_b.split()
    ia = "".join(t[0] for t in ta if t)
    ib = "".join(t[0] for t in tb if t)
    one = lambda toks, ini: len(toks) == 1 and len(ini) >= 2 and toks[0] == ini
    return float(one(tb, ia) or one(ta, ib))


def pair_row(a, b):
    """a, b: tuples in FIELDS order. Returns list of floats in NAMES order."""
    an, acore, askel, aaddr, ahnum, alast, _, araw = a
    bn, bcore, bskel, baddr, bhnum, blast, bscript, braw = b
    ca, cb = set(acore.split()), set(bcore.split())
    b_dom = bool(_DOMAIN.search(braw))
    dom_match = float(b_dom and acore.replace(" ", "") != "" and acore.replace(" ", "") in bn.replace(" ", ""))
    have_addr = bool(aaddr) and bool(baddr)
    ta, tb = set(aaddr.split()), set(baddr.split())
    na, nb = set(_NUM.findall(aaddr)), set(_NUM.findall(baddr))
    nan = np.nan
    ra, rb = _raw_core(araw), _raw_core(braw)
    la, lb = _NUM.findall(aaddr), _NUM.findall(baddr)
    if la and lb:
        fa, fb = int(la[0][:9]), int(lb[0][:9])
        num_first = [float(la[0] == lb[0]), abs(fa - fb) / max(fa, fb, 1)]
    else:
        num_first = [nan, nan]
    tail = _edit_sig(ra, rb) + [float(na == nb) if na and nb else nan, float(nb <= na) if na and nb else nan] + num_first
    return [
        fuzz.ratio(an, bn), fuzz.partial_ratio(an, bn), fuzz.token_set_ratio(an, bn), fuzz.token_sort_ratio(an, bn),
        fuzz.token_set_ratio(acore, bcore), fuzz.ratio(acore, bcore), JaroWinkler.similarity(acore, bcore),
        fuzz.token_set_ratio(askel, bskel), fuzz.ratio(askel, bskel),
        _jacc(ca, cb), _contain(ca, cb),
        float(bool(ca) and bool(cb) and acore.split()[0] == bcore.split()[0]),
        _acronym(acore, bcore), dom_match, float(b_dom),
        (min(len(acore), len(bcore)) / max(len(acore), len(bcore))) if acore and bcore else nan,
        len(ca), len(cb), _SCRIPT_ID.get(bscript, 0), float(bscript != "latin"),
        fuzz.ratio(aaddr, baddr) if have_addr else nan, fuzz.partial_ratio(aaddr, baddr) if have_addr else nan,
        fuzz.token_set_ratio(aaddr, baddr) if have_addr else nan, fuzz.token_sort_ratio(aaddr, baddr) if have_addr else nan,
        _jacc(ta, tb), (len(ta & tb) / len(tb)) if tb and ta else nan,
        _jacc(na, nb), (len(na & nb) / len(nb)) if nb and na else nan,
        float(ahnum == bhnum) if ahnum and bhnum else nan, float(bool(ahnum) and bool(bhnum)),
        float(alast == blast) if alast and blast else nan, float(not baddr),
        (min(len(aaddr), len(baddr)) / max(len(aaddr), len(baddr))) if have_addr else nan,
    ] + tail


def _chunk(args):
    A, B = args
    return np.array([pair_row(a, b) for a, b in zip(A, B)], dtype=np.float32)


def pair_features(A, B, procs=36, chunk=20000) -> np.ndarray:
    """A, B: lists of tuples (FIELDS order), aligned. Returns (n, len(NAMES)) float32."""
    jobs = [(A[i:i + chunk], B[i:i + chunk]) for i in range(0, len(A), chunk)]
    with Pool(procs) as p:
        parts = p.map(_chunk, jobs)
    return np.vstack(parts) if parts else np.zeros((0, len(NAMES)), np.float32)
