# eALS (element-wise ALS), Algorithm 1 of He et al. SIGIR'16, implemented in
# PySpark: the S^q/S^p caches are computed once per sweep and broadcast, and
# the per-user / per-item closed-form updates run in parallel via mapPartitions.
import functools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# force prints to flush immediately even when redirected to a log file (nohup, `>`)
print = functools.partial(print, flush=True)


@dataclass
class EALSConfig:
    n_factors: int = 32
    c0: float = 512.0          # overall weight of missing data, Eq. (8)
    alpha: float = 0.4         # popularity exponent, Eq. (8); alpha=0 -> uniform weighting (Hu et al. 2008)
    reg: float = 0.01          # lambda, L2 regularization
    w_obs: float = 1.0         # weight w_ui of observed (positive) entries
    n_iters: int = 10
    seed: int = 0
    n_partitions: int = 8


def item_confidence(item_pos_counts: np.ndarray, cfg: EALSConfig) -> np.ndarray:
    """c_i = c0 * f_i^alpha / sum_j f_j^alpha, Eq. (8). f_i is item i's share of interactions."""
    freq = item_pos_counts / item_pos_counts.sum()
    weighted = np.power(freq, cfg.alpha)
    return cfg.c0 * weighted / weighted.sum()


def init_factors(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (0.01 * rng.standard_normal(size=(n, k))).astype(np.float64)


def factor_cache(factors: np.ndarray, conf: np.ndarray = None) -> np.ndarray:
    """S = sum_i conf_i * f_i f_i^T. S^q uses conf=item_confidence (Eq. 10, 11); S^p uses conf=None, i.e. weight 1 -- the item's own c_i is applied separately in Eq. (13), see _update_items_chunk."""
    if conf is None:
        return factors.T @ factors
    return (factors * conf[:, None]).T @ factors


def _update_users_chunk(
    ids_chunk: np.ndarray,
    user_ratings: Dict[int, np.ndarray],
    P: np.ndarray,
    Q: np.ndarray,
    item_conf: np.ndarray,   # c_i, one value per item -- indexed by neighbor (Eq. 12)
    Sq: np.ndarray,
    reg: float,
    w_obs: float,
) -> np.ndarray:
    """Eq. (12): closed-form update of p_uf for every user in `ids_chunk`, iterating f=1..K."""
    K = P.shape[1]
    out = np.empty((len(ids_chunk), K), dtype=np.float64)
    for row, u in enumerate(ids_chunk):
        neighbors = user_ratings.get(u)
        pu = P[u].copy()
        if neighbors is None or len(neighbors) == 0:
            out[row] = pu
            continue
        Qu = Q[neighbors]                # |R_u| x K
        cu = item_conf[neighbors]         # |R_u|, c_i for each item u interacted with
        rhat = Qu @ pu
        for f in range(K):
            rhat_f = rhat - pu[f] * Qu[:, f]
            numer = np.sum((w_obs - (w_obs - cu) * rhat_f) * Qu[:, f]) - (pu @ Sq[:, f] - pu[f] * Sq[f, f])
            denom = np.sum((w_obs - cu) * Qu[:, f] ** 2) + Sq[f, f] + reg
            new_val = numer / denom if denom > 1e-12 else 0.0
            rhat = rhat_f + new_val * Qu[:, f]
            pu[f] = new_val
        out[row] = pu
    return out


def _update_items_chunk(
    ids_chunk: np.ndarray,
    item_ratings: Dict[int, np.ndarray],
    Q: np.ndarray,
    P: np.ndarray,
    item_conf: np.ndarray,   # c_i for the items in this chunk -- one scalar per row (Eq. 13)
    Sp: np.ndarray,
    reg: float,
    w_obs: float,
) -> np.ndarray:
    """Eq. (13): closed-form update of q_if for every item in `ids_chunk`, iterating f=1..K."""
    K = Q.shape[1]
    out = np.empty((len(ids_chunk), K), dtype=np.float64)
    for row, i in enumerate(ids_chunk):
        neighbors = item_ratings.get(i)
        qi = Q[i].copy()
        ci = item_conf[i]
        if neighbors is None or len(neighbors) == 0:
            out[row] = qi
            continue
        Pu = P[neighbors]                # |R_i| x K
        rhat = Pu @ qi
        for f in range(K):
            rhat_f = rhat - qi[f] * Pu[:, f]
            numer = np.sum((w_obs - (w_obs - ci) * rhat_f) * Pu[:, f]) - ci * (qi @ Sp[:, f] - qi[f] * Sp[f, f])
            denom = np.sum((w_obs - ci) * Pu[:, f] ** 2) + ci * Sp[f, f] + reg
            new_val = numer / denom if denom > 1e-12 else 0.0
            rhat = rhat_f + new_val * Pu[:, f]
            qi[f] = new_val
        out[row] = qi
    return out


class EALS:
    """Trains user/item latent factors with eALS on a Spark session.

    `train_pairs` is a list of (user, item) positive (implicit) interactions,
    users/items already re-indexed to [0, n_users) / [0, n_items).
    """

    def __init__(self, cfg: EALSConfig):
        self.cfg = cfg
        self.P: np.ndarray = None
        self.Q: np.ndarray = None
        self.item_conf: np.ndarray = None
        self.history: List[dict] = []

    def fit(self, sc, n_users: int, n_items: int, train_pairs: List[Tuple[int, int]], verbose: bool = True,
            eval_fn=None, eval_every: int = 1):
        """
        eval_fn, if given, is called as eval_fn(model, iteration) after every
        `eval_every`-th iteration; its return value (a dict) is merged into
        self.history. Used to trace HR@K/NDCG@K vs. iteration count (Figure 2
        of the paper) without retraining from scratch at each checkpoint.
        """
        cfg = self.cfg

        # adjacency lists (R_u, R_i) built once and reused every iteration
        user_ratings: Dict[int, np.ndarray] = {}
        item_ratings: Dict[int, np.ndarray] = {}
        item_pos_counts = np.ones(n_items)  # Laplace smoothing so unseen items keep c_i > 0
        _tmp_u: Dict[int, list] = {}
        _tmp_i: Dict[int, list] = {}
        for u, i in train_pairs:
            _tmp_u.setdefault(u, []).append(i)
            _tmp_i.setdefault(i, []).append(u)
            item_pos_counts[i] += 1
        user_ratings = {u: np.array(v, dtype=np.int64) for u, v in _tmp_u.items()}
        item_ratings = {i: np.array(v, dtype=np.int64) for i, v in _tmp_i.items()}

        self.item_conf = item_confidence(item_pos_counts, cfg)
        self.P = init_factors(n_users, cfg.n_factors, cfg.seed)
        self.Q = init_factors(n_items, cfg.n_factors, cfg.seed + 1)

        # glom() collapses each partition into one array, so the closed-form
        # update below runs once per partition instead of once per row
        user_chunks = sc.parallelize(np.arange(n_users), cfg.n_partitions).glom()
        item_chunks = sc.parallelize(np.arange(n_items), cfg.n_partitions).glom()

        ur_b = sc.broadcast(user_ratings)
        ir_b = sc.broadcast(item_ratings)
        ic_b = sc.broadcast(self.item_conf)

        for it in range(cfg.n_iters):
            # ---- update user factors: Algorithm 1, lines 4-11 ----------------------
            Sq = factor_cache(self.Q, self.item_conf)
            Q_b, Sq_b, P_local = sc.broadcast(self.Q), sc.broadcast(Sq), self.P

            def upd_users(ids_chunk, ur=ur_b, Qb=Q_b, Sqb=Sq_b, icb=ic_b, cfg=cfg, P=P_local):
                ids_chunk = np.asarray(ids_chunk)
                res = _update_users_chunk(ids_chunk, ur.value, P, Qb.value, icb.value, Sqb.value, cfg.reg, cfg.w_obs)
                return list(zip(ids_chunk.tolist(), res.tolist()))

            for uid, vec in user_chunks.flatMap(upd_users).collect():
                self.P[uid] = vec
            Q_b.unpersist(); Sq_b.unpersist()

            # ---- update item factors: Algorithm 1, lines 12-19 ----------------------
            Sp = factor_cache(self.P)
            P_b, Sp_b, Q_local = sc.broadcast(self.P), sc.broadcast(Sp), self.Q

            def upd_items(ids_chunk, ir=ir_b, Pb=P_b, Spb=Sp_b, icb=ic_b, cfg=cfg, Q=Q_local):
                ids_chunk = np.asarray(ids_chunk)
                res = _update_items_chunk(ids_chunk, ir.value, Q, Pb.value, icb.value, Spb.value, cfg.reg, cfg.w_obs)
                return list(zip(ids_chunk.tolist(), res.tolist()))

            for iid, vec in item_chunks.flatMap(upd_items).collect():
                self.Q[iid] = vec
            P_b.unpersist(); Sp_b.unpersist()

            # only keep a history entry if we actually logged something this round
            record = {"iter": it + 1}
            if verbose:
                record["loss"] = self.compute_loss(train_pairs)
                print(f"[eALS] iter {it + 1}/{cfg.n_iters}  loss={record['loss']:.4f}")
            if eval_fn is not None and (it + 1) % eval_every == 0:
                record.update(eval_fn(self, it + 1))
            if len(record) > 1:
                self.history.append(record)

        ur_b.unpersist(); ir_b.unpersist(); ic_b.unpersist()
        return self

    def compute_loss(self, train_pairs) -> float:
        """Objective L, Eq. (14), evaluated vectorized in NumPy on the driver."""
        cfg = self.cfg
        Sq = factor_cache(self.Q, self.item_conf)
        missing_term = np.einsum("uk,kl,ul->u", self.P, Sq, self.P).sum()

        users = np.fromiter((u for u, _ in train_pairs), dtype=np.int64, count=len(train_pairs))
        items = np.fromiter((i for _, i in train_pairs), dtype=np.int64, count=len(train_pairs))
        rhat = np.einsum("nk,nk->n", self.P[users], self.Q[items])
        obs_term = cfg.w_obs * np.sum((1.0 - rhat) ** 2)
        correction = np.sum(self.item_conf[items] * rhat ** 2)

        reg_term = cfg.reg * (np.sum(self.P ** 2) + np.sum(self.Q ** 2))
        return float(obs_term + (missing_term - correction) + reg_term)

    def predict(self, u: int, i: int) -> float:
        return float(self.P[u] @ self.Q[i])

    def score_all_items(self, u: int) -> np.ndarray:
        return self.Q @ self.P[u]
