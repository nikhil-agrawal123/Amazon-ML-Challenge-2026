"""Loading raw TSVs and the normalised parquet cache."""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
CACHE = ROOT / "cache"


def read_tsv(split: str, name: str) -> pl.DataFrame:
    """Read dataset/<split>/<split>_<name>.tsv as all-string, empty -> ''."""
    return pl.read_csv(
        DATA / split / f"{split}_{name}.tsv",
        separator="\t",
        quote_char=None,
        infer_schema=False,
    ).fill_null("")


def load_sources(split: str) -> pl.DataFrame:
    """All three sources stacked, with a `src` column (1/2/3)."""
    return pl.concat(
        [read_tsv(split, f"source{i}").with_columns(pl.lit(i, pl.Int8).alias("src")) for i in (1, 2, 3)]
    )


def load_gt_pairs() -> pl.DataFrame:
    """Train ground truth exploded to one row per (s1, match) pair; singletons dropped."""
    gt = read_tsv("train", "ground_truth")
    return (
        gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m")
        .filter(pl.col("m") != "")
        .select(pl.col("source1_entity_id").alias("id1"), pl.col("m").alias("id2"))
    )


def load_norm(split: str) -> pl.DataFrame:
    """Normalised records (built by scripts/build_norm_cache.py)."""
    return pl.read_parquet(CACHE / f"{split}_norm.parquet")
