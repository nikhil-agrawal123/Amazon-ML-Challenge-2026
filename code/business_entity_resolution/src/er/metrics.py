"""Blocking / matching metrics. Macro F0.5 per S1 entity, singletons count 1.0 on empty prediction."""
import polars as pl


def f05_per_entity(k: pl.Expr, found: pl.Expr, npred: pl.Expr) -> pl.Expr:
    p = found / npred
    r = found / k
    f = 1.25 * p * r / (0.25 * p + r)
    return (pl.when(k == 0).then((npred == 0).cast(pl.Float64))
            .when(found == 0).then(0.0)
            .otherwise(f))


def blocking_report(cands: pl.DataFrame, truth: pl.DataFrame, queries: pl.DataFrame, name: str) -> dict:
    """cands/truth: (r1, r2); queries: (r1, country). F05_ceiling = perfect matcher on these candidates."""
    c = cands.select("r1", "r2").unique().join(queries.select("r1"), on="r1")
    t = truth.join(queries.select("r1"), on="r1")
    hit = c.join(t, on=["r1", "r2"])
    per = queries.join(t.group_by("r1").len().rename({"len": "k"}), on="r1", how="left") \
                 .join(hit.group_by("r1").len().rename({"len": "found"}), on="r1", how="left") \
                 .join(c.group_by("r1").len().rename({"len": "ncand"}), on="r1", how="left").fill_null(0)
    per = per.with_columns(f05_per_entity(pl.col("k"), pl.col("found"), pl.col("found")).fill_nan(1.0).alias("f"))
    row = {"scheme": name, "pair_recall": round(hit.height / max(t.height, 1), 4), "F05_ceiling": round(per["f"].mean(), 4)}
    for cc in sorted(queries["country"].unique().to_list()):
        row[f"F05_{cc}"] = round(per.filter(pl.col("country") == cc)["f"].mean(), 4)
    row["cands_mean"] = round(per["ncand"].mean(), 1)
    return row
