# Fast Matrix Factorization for Online Recommendation — PySpark eALS

A PySpark implementation and experimental evaluation of **eALS** (element-wise
Alternating Least Squares), the implicit-feedback matrix factorization
algorithm from:

> Xiangnan He, Hanwang Zhang, Min-Yen Kan, Tat-Seng Chua. *Fast Matrix
> Factorization for Online Recommendation with Implicit Feedback.* SIGIR 2016.
> ([arXiv:1708.05024](https://arxiv.org/abs/1708.05024))

The paper describes an efficient, embarrassingly-parallel ALS variant but
ships no MapReduce/Spark implementation. This project provides one (RDD-based,
`src/eals.py`), evaluates it on the Amazon "Movies & TV" review dataset used
in the paper's own experiments, and studies its scalability.

## Layout

```
src/
  eals.py             eALS core algorithm (Algorithm 1), PySpark RDD-based
  baselines.py         Spark MLlib ALS reference baseline
  evaluate.py           HR@K / NDCG@K leave-one-out evaluation
  preprocess.py          raw Amazon reviews -> filtered implicit-feedback train/test parquet
  run_experiments.py      weighting / convergence / scalability experiments -> results/*.csv
  plot_results.py           turns results/*.csv into the figures used in the report
tests/
  test_eals_synthetic.py   correctness check on a small synthetic dataset
notebooks/
  eals_demo.ipynb            small-data, end-to-end, commented notebook (Appendix requirement)
data/
  raw/                  downloaded Amazon ratings CSV (not committed)
  processed/              filtered/reindexed train & test parquet (not committed)
results/
  *.csv                    experiment outputs
  figures/*.png              figures used throughout the report
latex/
  report.tex, chapters/, report.pdf   the full report (see Report below)
```

The uniform-weighting ablation (Hu et al.'s objective) is not a separate
module: it is just `EALSConfig(alpha=0.0)`, used directly in
`run_experiments.py` since it isolates the effect of the popularity
weighting while keeping the same eALS solver.

## Report

The full report (problem description, algorithm derivation, experiments,
discussion, and a code appendix) lives in `latex/` and builds with:

```bash
cd latex && make
```

which produces `latex/report.pdf`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install pyspark numpy pandas scipy matplotlib jupyter nbformat pyarrow
```

Requires a JDK (Spark runs on the JVM); tested with Java 17/21 and PySpark 3.5.

The raw and filtered dataset (`data/raw/`, `data/processed/`) are tracked
with [Git LFS](https://git-lfs.com) rather than checked out on clone by
default. Run `git lfs pull` after cloning if you want them locally instead
of regenerating them (see below).

## Reproducing the experiments

```bash
# 1. Download & filter the dataset (Amazon "Movies and TV" ratings, as in the paper)
curl -o data/raw/ratings_Movies_and_TV.csv \
  https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv
python src/preprocess.py --raw data/raw/ratings_Movies_and_TV.csv --out data/processed --min-interactions 10

# 2. Run the experimental analysis (weighting study, convergence, scalability)
python src/run_experiments.py --data data/processed --out results

# 3. Sanity-check the algorithm on synthetic data
python tests/test_eals_synthetic.py

# 4. Small-data working demo (Appendix requirement)
jupyter nbconvert --to notebook --execute --inplace notebooks/eals_demo.ipynb
```

## Dataset

We use the same Amazon "Movies and TV" implicit-feedback dataset as the
paper (reviews binarized to a 0/1 interaction), applying the same k-core
filter (users/items with fewer than 10 interactions removed, applied
iteratively to convergence -- `preprocess.py` explicitly checks and reports
that every remaining user/item has >= 10 interactions). The Yelp Challenge
dataset used in the original paper is no longer distributed by Yelp in that
form, so this project sticks to Amazon Movies, the second dataset the paper
reports on.

|                | Paper (Amazon) | This project |
|----------------|---------------:|-------------:|
| Users          |        117,176 |        33,326 |
| Items          |         75,389 |        21,901 |
| Interactions   |      5,020,705 |       958,986 |
| Sparsity       |         99.94% |        99.87% |

The smaller scale is expected: the CSV hosted today by SNAP is a different
snapshot than the one the authors filtered in 2016, so raw interaction
counts differ before any filtering is even applied.

## Notes on deviations from the paper's exact protocol

- **Ranking cutoff during evaluation**: the paper ranks the held-out item
  against the *entire* item catalog. For a scalability study that retrains
  the model dozens of times, we default to ranking against `n_negatives`
  sampled negatives (`evaluate.evaluate_sampled`), the standard practical
  relaxation used in follow-up work by the same authors (NCF, WWW'17). The
  paper's exact full-catalog protocol is also implemented
  (`evaluate.evaluate_full`) and used in the small-data notebook.
- **Matrix-inversion ALS baseline**: the paper's `ALS` baseline uses a
  vector-wise solver with O(K^3) matrix inversion. We compare against
  Spark MLlib's own implicit-feedback `ALS` instead of hand-rolling that
  inversion in Spark, since MLlib's is a widely used, independently
  engineered reference point with the same theoretical objective (Hu et
  al. 2008).

## License

This project is licensed under the MIT License — see
[LICENSE](https://github.com/sepanta007/Matrix_Factorization/blob/master/LICENSE).
