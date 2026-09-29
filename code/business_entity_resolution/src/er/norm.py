"""Country-agnostic string normalisation for names/addresses.

Everything here is pure-python per string; use `normalise_frame` to run it
over millions of rows with a process pool.
"""
import re
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REPEAT = re.compile(r"([a-z])\1+")
_VOWELS = re.compile(r"[aeiouy]")
_DIGITS = re.compile(r"\d+")

# Legal / generic words stripped from names for the "core" name. Includes
# transliterated forms seen in Indic-script names (praivet, limited after
# repeat-collapse) and French forms (sarl, sas, ...), since France is test-only.
LEGAL = {
    "pvt", "private", "praivet", "privet", "ltd", "limited", "limted", "lmtd", "llc", "l", "inc", "incorporated",
    "corp", "corporation", "co", "company", "llp", "lp", "plc", "pc", "pllc", "public", "the", "and", "of",
    "group", "groupe", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "et", "de", "des", "du", "la", "le",
    "dba", "india", "indian", "usa", "us", "america", "france", "com", "www", "net", "org", "in",
}

SCRIPTS = [
    ("devanagari", 0x0900, 0x097F), ("bengali", 0x0980, 0x09FF), ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF), ("oriya", 0x0B00, 0x0B7F), ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F), ("kannada", 0x0C80, 0x0CFF), ("malayalam", 0x0D00, 0x0D7F),
]


def script_of(s: str) -> str:
    for ch in s:
        o = ord(ch)
        if o >= 0x0900:
            for name, lo, hi in SCRIPTS:
                if lo <= o <= hi:
                    return name
            return "other"
    return "latin"


def basic(s: str) -> str:
    """ascii-fold, lowercase, & -> and, punctuation -> space, collapse repeated letters."""
    s = unidecode(s).lower().replace("&", " and ")
    s = _NON_ALNUM.sub(" ", s)
    s = _REPEAT.sub(r"\1", s)
    return " ".join(s.split())


def skel_token(t: str) -> str:
    """Consonant skeleton: keep first char, drop later vowels. ph->f, w->v."""
    if not t or t[0].isdigit():
        return t
    t = t.replace("ph", "f").replace("w", "v").replace("z", "j")
    return t[0] + _VOWELS.sub("", t[1:])


def name_fields(raw: str) -> tuple[str, str, str]:
    n = basic(raw)
    core = [t for t in n.split() if t not in LEGAL]
    skel = [skel_token(t) for t in core]
    return n, " ".join(core), " ".join(skel)


def addr_fields(raw: str) -> tuple[str, str, str]:
    """(normalised address, first number, last comma-part normalised)."""
    n = basic(raw)
    m = _DIGITS.search(n)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    last = basic(parts[-1]) if parts else ""
    return n, (m.group(0) if m else ""), last


def _row(args):
    name, addr = args
    return (*name_fields(name), *addr_fields(addr), script_of(name), script_of(addr))


COLS = ["name_n", "name_core", "name_skel", "addr_n", "hnum", "addr_last", "name_script", "addr_script"]


def normalise_frame(df: pl.DataFrame, procs: int = 36) -> pl.DataFrame:
    items = list(zip(df["business_name"].to_list(), df["business_address"].to_list()))
    with Pool(procs) as p:
        rows = p.map(_row, items, chunksize=20000)
    cols = list(zip(*rows))
    return df.with_columns([pl.Series(c, v) for c, v in zip(COLS, cols)])
