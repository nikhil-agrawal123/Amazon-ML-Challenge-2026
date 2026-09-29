"""Build cache/{train,test}_norm.parquet: raw fields + normalised columns."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from er.io import CACHE, load_sources  # noqa: E402
from er.norm import normalise_frame  # noqa: E402

if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    for split in ("train", "test"):
        t = time.time()
        df = load_sources(split)
        df = normalise_frame(df)
        df.write_parquet(CACHE / f"{split}_norm.parquet")
        print(f"{split}: {df.shape} in {time.time() - t:.0f}s", flush=True)
