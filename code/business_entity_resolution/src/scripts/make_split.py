"""Fixed train/val split over S1 entities -> cache/split.parquet (entity_id, fold 0..9).

fold 0 = validation for every model in the project (bi-encoder and matcher).
Hash-based so it is reproducible and independent of row order.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from er.io import CACHE, read_tsv  # noqa: E402

if __name__ == "__main__":
    s1 = read_tsv("train", "source1").select("entity_id", "country")
    s1 = s1.with_columns((pl.col("entity_id").hash(seed=2026) % 10).cast(pl.Int8).alias("fold"))
    s1.select("entity_id", "fold").write_parquet(CACHE / "split.parquet")
    print(s1.group_by("country", "fold").len().sort("country", "fold"))
