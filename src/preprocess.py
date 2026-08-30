# Filters the raw Amazon reviews down to a 10-core implicit-feedback dataset
# and writes the reindexed train/test parquet files used by every experiment.
import argparse
import functools
import shutil

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# force prints to flush immediately even when redirected to a log file (nohup, `>`)
print = functools.partial(print, flush=True)


def iterative_core_filter(spark, df, min_interactions, tmp_dir, max_rounds=50):
    """Repeatedly drop users/items below the interaction threshold until stable (k-core filtering).
    Each round is materialized to parquet and reloaded from disk rather than just `.cache()`-d:
    caching alone keeps extending the logical query plan every round (dedup -> N joins), and Spark's
    adaptive-execution plan-string logging on that ever-growing plan is what triggered the driver
    OutOfMemoryError seen at round ~6. Writing to disk truncates the lineage every round."""
    before = df.count()
    converged = False
    for rnd in range(max_rounds):
        keep_users = df.groupBy("user_raw").count().filter(F.col("count") >= min_interactions).select("user_raw")
        keep_items = df.groupBy("item_raw").count().filter(F.col("count") >= min_interactions).select("item_raw")
        filtered = df.join(keep_users, "user_raw").join(keep_items, "item_raw")

        round_path = f"{tmp_dir}/round_{rnd}.parquet"
        filtered.write.mode("overwrite").parquet(round_path)
        df = spark.read.parquet(round_path)

        after = df.count()
        print(f"  core-filter round {rnd + 1}: {before} -> {after} interactions")
        if after == before:
            converged = True
            break
        before = after
    if not converged:
        print(f"  WARNING: core-filter did not fully converge within {max_rounds} rounds "
              f"(every remaining user/item still had >= {min_interactions} interactions in the last "
              f"round is NOT guaranteed) -- increase max_rounds if you need an exact k-core.")
    else:
        print(f"  core-filter converged: every remaining user and item has >= {min_interactions} interactions.")
    return df


def main(raw_path, out_dir, min_interactions, max_rounds=50):
    spark = (
        SparkSession.builder.appName("eals-preprocess")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )

    raw = spark.read.csv(
        raw_path,
        schema="user_raw STRING, item_raw STRING, rating DOUBLE, ts LONG",
    )

    tmp_dir = f"{out_dir}/_tmp_corefilter"
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Implicit feedback: any review/rating becomes a positive interaction (rui = 1).
    # Duplicate (user, item) pairs collapse to their most recent timestamp.
    dedup_path = f"{tmp_dir}/dedup.parquet"
    raw.groupBy("user_raw", "item_raw").agg(F.max("ts").alias("ts")).write.mode("overwrite").parquet(dedup_path)
    interactions = spark.read.parquet(dedup_path)
    print("raw deduped interactions:", interactions.count())

    interactions = iterative_core_filter(spark, interactions, min_interactions, tmp_dir, max_rounds=max_rounds)

    # row_number() starts at 1, subtract 1 so ids land in [0, n) like eals.py expects
    user_index = (
        interactions.select("user_raw").distinct()
        .withColumn("user", F.row_number().over(Window.orderBy("user_raw")) - 1)
    )
    item_index = (
        interactions.select("item_raw").distinct()
        .withColumn("item", F.row_number().over(Window.orderBy("item_raw")) - 1)
    )

    interactions = (
        interactions.join(user_index, "user_raw")
        .join(item_index, "item_raw")
        .select("user", "item", "ts")
    )

    # Offline / leave-one-out protocol (Section 5.1): each user's chronologically
    # latest interaction is held out for testing, the rest is training data.
    w = Window.partitionBy("user").orderBy(F.col("ts").desc())
    ranked = interactions.withColumn("rank", F.row_number().over(w))
    test = ranked.filter(F.col("rank") == 1).select("user", "item")
    train = ranked.filter(F.col("rank") > 1).select("user", "item")

    interactions.select("user", "item", "ts").write.mode("overwrite").parquet(f"{out_dir}/interactions.parquet")
    train.write.mode("overwrite").parquet(f"{out_dir}/train.parquet")
    test.write.mode("overwrite").parquet(f"{out_dir}/test.parquet")
    user_index.write.mode("overwrite").parquet(f"{out_dir}/user_index.parquet")
    item_index.write.mode("overwrite").parquet(f"{out_dir}/item_index.parquet")

    n_users = user_index.count()
    n_items = item_index.count()
    n_train = train.count()
    n_test = test.count()
    print(f"users={n_users} items={n_items} train={n_train} test={n_test} "
          f"sparsity={1 - (n_train + n_test) / (n_users * n_items):.4%}")

    spark.stop()
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/ratings_Movies_and_TV.csv")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--min-interactions", type=int, default=10)
    args = parser.parse_args()
    main(args.raw, args.out, args.min_interactions)
