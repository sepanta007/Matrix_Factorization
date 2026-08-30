# Runs the weighting sweep, the eALS-vs-uniform convergence comparison, and
# the scalability study, each writing a CSV to results/.
import argparse
import functools
import os
import time
from typing import Dict, List

import numpy as np
from pyspark.sql import SparkSession

# stdout is block-buffered when redirected to a file/log (nohup, `>`), so plain
# print() calls sit unseen in a buffer until it fills or the process exits --
# making a long background run look stuck. Force a flush on every print instead.
print = functools.partial(print, flush=True)

from eals import EALS, EALSConfig
from baselines import fit_spark_mllib_als
from evaluate import evaluate_sampled


def load_data(spark, data_dir):
    train_df = spark.read.parquet(f"{data_dir}/train.parquet")
    test_df = spark.read.parquet(f"{data_dir}/test.parquet")
    n_users = spark.read.parquet(f"{data_dir}/user_index.parquet").count()
    n_items = spark.read.parquet(f"{data_dir}/item_index.parquet").count()

    train_pairs = [(r["user"], r["item"]) for r in train_df.collect()]
    test_pairs = [(r["user"], r["item"]) for r in test_df.collect()]

    train_user_items: Dict[int, set] = {}
    for u, i in train_pairs:
        train_user_items.setdefault(u, set()).add(i)

    return n_users, n_items, train_pairs, test_pairs, train_user_items


def weighting_study(sc, n_users, n_items, train_pairs, test_pairs, train_user_items, out_dir):
    """Grid over c0 (alpha=0) then over alpha (c0 fixed at the best value) -- mirrors Figure 1."""
    rows = []
    for c0 in [8, 32, 128, 512, 2048]:
        cfg = EALSConfig(n_factors=32, c0=c0, alpha=0.0, reg=0.01, n_iters=10, n_partitions=8)
        model = EALS(cfg).fit(sc, n_users, n_items, train_pairs, verbose=False)
        metrics = evaluate_sampled(model.P, model.Q, test_pairs, train_user_items, k=10)
        rows.append({"sweep": "c0", "c0": c0, "alpha": 0.0, **metrics})
        print(rows[-1])

    best_c0 = max([r for r in rows if r["sweep"] == "c0"], key=lambda r: r["HR@10"])["c0"]
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]:
        cfg = EALSConfig(n_factors=32, c0=best_c0, alpha=alpha, reg=0.01, n_iters=10, n_partitions=8)
        model = EALS(cfg).fit(sc, n_users, n_items, train_pairs, verbose=False)
        metrics = evaluate_sampled(model.P, model.Q, test_pairs, train_user_items, k=10)
        rows.append({"sweep": "alpha", "c0": best_c0, "alpha": alpha, **metrics})
        print(rows[-1])

    _write_csv(rows, f"{out_dir}/weighting_study.csv")
    return best_c0


def method_comparison(spark, sc, n_users, n_items, train_pairs, test_pairs, train_user_items, out_dir, best_c0):
    """Convergence (HR@10/NDCG@10 vs. iteration) for popularity-aware vs. uniform-weight eALS -- mirrors Figure 2."""
    rows = []
    n_iters = 15

    def make_eval_fn(name, t_start):
        # runs every iteration, so elapsed_seconds here bundles eval time in
        # with training time -- not directly comparable to scalability_study's numbers
        def eval_fn(model, it):
            metrics = evaluate_sampled(model.P, model.Q, test_pairs, train_user_items, k=10)
            row = {"method": name, "iter": it, "elapsed_seconds": time.time() - t_start[0], **metrics}
            rows.append(row)
            return {}
        return eval_fn

    for name, alpha in [("eALS (popularity-aware)", 0.4), ("eALS (uniform alpha=0)", 0.0)]:
        cfg = EALSConfig(n_factors=32, c0=best_c0, alpha=alpha, reg=0.01, n_iters=n_iters, n_partitions=8)
        t_start = [time.time()]
        EALS(cfg).fit(sc, n_users, n_items, train_pairs, verbose=False, eval_fn=make_eval_fn(name, t_start))
        print(name, rows[-1])

    _write_csv(rows, f"{out_dir}/method_comparison.csv")


def scalability_study(sc, n_users, n_items, train_pairs, test_pairs, train_user_items, out_dir, best_c0):
    rows = []
    # (a) training time per iteration vs. number of latent factors K
    for K in [8, 16, 32, 64, 128]:
        cfg = EALSConfig(n_factors=K, c0=best_c0, alpha=0.4, reg=0.01, n_iters=3, n_partitions=8)
        t0 = time.time()
        model = EALS(cfg).fit(sc, n_users, n_items, train_pairs, verbose=False)
        elapsed = (time.time() - t0) / cfg.n_iters
        metrics = evaluate_sampled(model.P, model.Q, test_pairs, train_user_items, k=10)
        rows.append({"axis": "K", "K": K, "n_partitions": 8, "seconds_per_iter": elapsed, **metrics})
        print(rows[-1])

    # (b) training time per iteration vs. number of Spark partitions (parallelism)
    for P_ in [1, 2, 4, 8, 16]:
        cfg = EALSConfig(n_factors=32, c0=best_c0, alpha=0.4, reg=0.01, n_iters=3, n_partitions=P_)
        t0 = time.time()
        model = EALS(cfg).fit(sc, n_users, n_items, train_pairs, verbose=False)
        elapsed = (time.time() - t0) / cfg.n_iters
        rows.append({"axis": "n_partitions", "K": 32, "n_partitions": P_, "seconds_per_iter": elapsed})
        print(rows[-1])

    # (c) training time per iteration vs. training-set size (data scalability)
    rng = np.random.RandomState(0)
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        idx = rng.choice(len(train_pairs), size=int(len(train_pairs) * frac), replace=False)
        sub_pairs = [train_pairs[i] for i in idx]
        cfg = EALSConfig(n_factors=32, c0=best_c0, alpha=0.4, reg=0.01, n_iters=3, n_partitions=8)
        t0 = time.time()
        EALS(cfg).fit(sc, n_users, n_items, sub_pairs, verbose=False)
        elapsed = (time.time() - t0) / cfg.n_iters
        rows.append({"axis": "data_frac", "K": 32, "n_partitions": 8, "data_frac": frac,
                      "n_train": len(sub_pairs), "seconds_per_iter": elapsed})
        print(rows[-1])

    _write_csv(rows, f"{out_dir}/scalability_study.csv")


def spark_mllib_reference(spark, train_pairs, test_pairs, train_user_items, n_users, n_items, out_dir):
    rows = []
    for K in [8, 16, 32, 64, 128]:
        t0 = time.time()
        model = fit_spark_mllib_als(spark, train_pairs, n_factors=K, reg=0.01, n_iters=10)
        elapsed = (time.time() - t0) / 10
        # MLlib hands back factors as a DataFrame; pull them into the same
        # dense P/Q layout evaluate_sampled expects
        P = np.zeros((n_users, K)); Q = np.zeros((n_items, K))
        for r in model.userFactors.collect():
            P[r["id"]] = r["features"]
        for r in model.itemFactors.collect():
            Q[r["id"]] = r["features"]
        metrics = evaluate_sampled(P, Q, test_pairs, train_user_items, k=10)
        rows.append({"K": K, "seconds_per_iter": elapsed, **metrics})
        print(rows[-1])
    _write_csv(rows, f"{out_dir}/spark_mllib_als.csv")


def _write_csv(rows: List[dict], path: str):
    if not rows:
        return
    import csv
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="results")
    parser.add_argument("--skip-weighting", action="store_true")
    parser.add_argument("--skip-mllib", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spark = SparkSession.builder.appName("eals-experiments").config("spark.driver.memory", "6g").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    sc.addPyFile("src/eals.py")

    n_users, n_items, train_pairs, test_pairs, train_user_items = load_data(spark, args.data)
    print(f"n_users={n_users} n_items={n_items} n_train={len(train_pairs)} n_test={len(test_pairs)}")

    best_c0 = 512.0
    if not args.skip_weighting:
        best_c0 = weighting_study(sc, n_users, n_items, train_pairs, test_pairs, train_user_items, args.out)

    method_comparison(spark, sc, n_users, n_items, train_pairs, test_pairs, train_user_items, args.out, best_c0)
    scalability_study(sc, n_users, n_items, train_pairs, test_pairs, train_user_items, args.out, best_c0)

    if not args.skip_mllib:
        spark_mllib_reference(spark, train_pairs, test_pairs, train_user_items, n_users, n_items, args.out)

    spark.stop()


if __name__ == "__main__":
    main()
