# HR@k / NDCG@k for the leave-one-out protocol. evaluate_full ranks against
# the whole catalog (paper's exact protocol); evaluate_sampled ranks against
# n_negatives random items instead, used for the large-scale experiments.
from typing import Dict, List, Tuple

import numpy as np


def _hr_ndcg_at_k(rank: int, k: int) -> Tuple[float, float]:
    # rank is 0-indexed, so a hit at the very top scores log2(2) = 1
    if rank < 0 or rank >= k:
        return 0.0, 0.0
    return 1.0, 1.0 / np.log2(rank + 2)


def evaluate_sampled(
    P: np.ndarray,
    Q: np.ndarray,
    test_pairs: List[Tuple[int, int]],
    train_user_items: Dict[int, set],
    k: int = 10,
    n_negatives: int = 99,
    seed: int = 0,
) -> Dict[str, float]:
    """HR@k / NDCG@k, ranking each held-out positive against `n_negatives` sampled negatives."""
    rng = np.random.RandomState(seed)
    n_items = Q.shape[0]
    hrs, ndcgs = [], []
    for u, pos_item in test_pairs:
        seen = train_user_items.get(u, set())
        negatives = []
        while len(negatives) < n_negatives:  # rejection sampling, fine at this n_items
            cand = rng.randint(0, n_items)
            if cand != pos_item and cand not in seen:
                negatives.append(cand)
        candidates = np.array([pos_item] + negatives)
        scores = Q[candidates] @ P[u]
        rank = int(np.sum(scores > scores[0]))  # how many candidates score above the positive
        hr, ndcg = _hr_ndcg_at_k(rank, k)
        hrs.append(hr)
        ndcgs.append(ndcg)
    return {"HR@%d" % k: float(np.mean(hrs)), "NDCG@%d" % k: float(np.mean(ndcgs)), "n_test": len(test_pairs)}


def evaluate_full(
    P: np.ndarray,
    Q: np.ndarray,
    test_pairs: List[Tuple[int, int]],
    train_user_items: Dict[int, set],
    k: int = 100,
) -> Dict[str, float]:
    """Exact paper protocol: rank the held-out item against the *entire* catalog
    (excluding items already seen in training). Only tractable for small M, N."""
    hrs, ndcgs = [], []
    for u, pos_item in test_pairs:
        scores = Q @ P[u]
        seen = train_user_items.get(u, set())
        if seen:
            scores = scores.copy()
            scores[list(seen)] = -np.inf
        rank = int(np.sum(scores > scores[pos_item]))
        hr, ndcg = _hr_ndcg_at_k(rank, k)
        hrs.append(hr)
        ndcgs.append(ndcg)
    return {"HR@%d" % k: float(np.mean(hrs)), "NDCG@%d" % k: float(np.mean(ndcgs)), "n_test": len(test_pairs)}
