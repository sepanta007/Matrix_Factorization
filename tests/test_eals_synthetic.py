# Two disjoint user/item communities; a correct model should rank each
# user's own community above the other one and the loss should decrease.
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from pyspark.sql import SparkSession

from eals import EALS, EALSConfig


def make_two_community_data(n_users=20, n_items=20, seed=0):
    rng = np.random.RandomState(seed)
    pairs = []
    # community A: users 0-9 interact with items 0-9
    for u in range(10):
        items = rng.choice(10, size=5, replace=False)
        pairs += [(u, int(i)) for i in items]
    # community B: users 10-19 interact with items 10-19
    for u in range(10, 20):
        items = rng.choice(range(10, 20), size=5, replace=False)
        pairs += [(u, int(i)) for i in items]
    return pairs, n_users, n_items


def main():
    spark = SparkSession.builder.master("local[4]").appName("eals-test").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    sc.addPyFile(os.path.join(os.path.dirname(__file__), "..", "src", "eals.py"))

    pairs, n_users, n_items = make_two_community_data()
    cfg = EALSConfig(n_factors=8, c0=32.0, alpha=0.4, reg=0.05, n_iters=15, n_partitions=4)
    model = EALS(cfg).fit(sc, n_users, n_items, pairs)

    losses = [h["loss"] for h in model.history]
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    scores_u0 = model.score_all_items(0)
    within = scores_u0[:10].mean()
    across = scores_u0[10:].mean()
    print(f"user 0: within-community avg score={within:.4f}  across-community avg score={across:.4f}")
    assert within > across, "eALS failed to separate the two communities"

    print("OK: loss history:", [f"{l:.2f}" for l in losses])
    print("PASSED")
    spark.stop()


if __name__ == "__main__":
    main()
