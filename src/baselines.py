# Spark MLlib's own implicit ALS, used as an external reference point.
from typing import List, Tuple

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS as SparkALS


def fit_spark_mllib_als(
    spark: SparkSession,
    train_pairs: List[Tuple[int, int]],
    n_factors: int = 32,
    reg: float = 0.01,
    n_iters: int = 10,
    alpha: float = 1.0,  # MLlib's own confidence scaling, unrelated to eALS's popularity alpha
):
    """Spark's own implicit-feedback ALS (Hu et al. 2008), for reference."""
    df = spark.createDataFrame(train_pairs, ["user", "item"]).withColumn("rating", F.lit(1.0))
    als = SparkALS(
        userCol="user", itemCol="item", ratingCol="rating",
        rank=n_factors, maxIter=n_iters, regParam=reg, alpha=alpha,
        implicitPrefs=True, coldStartStrategy="drop", seed=0,
    )
    return als.fit(df)
