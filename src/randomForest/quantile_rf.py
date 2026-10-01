"""Quantile regression forest (Meinshausen 2006) on top of sklearn's
RandomForestRegressor.

Unlike a point RandomForestRegressor (which predicts the *mean* of each leaf),
this class predicts arbitrary quantiles from the empirical distribution of
training targets that fall in the same leaves as each test point.

Weights: for a test point x and tree t, every training sample sharing x's leaf
gets weight 1/(K * |leaf|). Accumulating over the K trees gives a per-test
empirical distribution over the training targets; quantiles are read off that
distribution. This is the standard QRF weighting scheme.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class QuantileRegressionForest:
    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 2,
        max_features=None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.y_train_ = np.asarray(y, dtype=float)
        self.leaf_train_ = self.rf.apply(X)          # (n_train, n_estimators)
        self._order = np.argsort(self.y_train_)
        self._y_sorted = self.y_train_[self._order]
        return self

    def predict(self, X, quantiles=0.5):
        """Return empirical quantile(s) of the target for each row of X.

        quantiles: scalar or array-like. Scalar -> shape (n_samples,);
        array-like of length m -> shape (n_samples, m).

        Sparse + memory-bounded: instead of a dense (n_test, n_train) weight
        matrix (which is O(n_test * n_train) memory and blows up for large test
        sets), weights are accumulated leaf-by-leaf via precomputed per-tree
        sorted leaf groupings, processing the test set in chunks. The weight of
        a training point sharing leaf l of tree t is 1/(K * |l|), identical to
        the naive formulation.
        """
        leaf_test = self.rf.apply(X)                 # (n_test, n_estimators)
        n_test = leaf_test.shape[0]
        n_train = self.leaf_train_.shape[0]
        K = self.rf.n_estimators

        q = np.atleast_1d(np.asarray(quantiles, dtype=float))
        out = np.empty((n_test, len(q)))

        # Precompute per-tree sorted leaf groupings (once per predict call).
        sorted_leaf = []
        sorted_row = []
        for t in range(K):
            lt = self.leaf_train_[:, t]
            sidx = np.argsort(lt, kind="stable")
            sorted_leaf.append(lt[sidx])
            sorted_row.append(sidx)

        order = self._order
        y_sorted = self._y_sorted

        # Cap chunk weights footprint at ~200 MB: m * n_train <= 25e6.
        chunk = max(1, int(25_000_000 // max(1, n_train)))
        for start in range(0, n_test, chunk):
            end = min(start + chunk, n_test)
            m = end - start
            weights = np.zeros((m, n_train), dtype=np.float64)
            for t in range(K):
                le = leaf_test[start:end, t]
                ls = sorted_leaf[t]
                rs = sorted_row[t]
                lo = np.searchsorted(ls, le, side="left")
                hi = np.searchsorted(ls, le, side="right")
                counts = hi - lo
                total = int(counts.sum())
                if total == 0:
                    continue
                pidx = np.repeat(np.arange(m), counts)
                starts_rep = np.repeat(lo, counts)
                off = np.arange(total) - np.repeat(
                    np.cumsum(counts) - counts, counts
                )
                row = rs[starts_rep + off]
                wval = np.repeat(
                    1.0 / (K * np.maximum(counts, 1).astype(np.float64)), counts
                )
                np.add.at(weights, (pidx, row), wval)

            row_sum = weights.sum(axis=1, keepdims=True)
            weights = weights / np.where(row_sum == 0.0, 1.0, row_sum)

            for j in range(m):
                c = np.cumsum(weights[j][order])
                idx = np.searchsorted(c, q)
                idx = np.clip(idx, 0, n_train - 1)
                out[start + j] = y_sorted[idx]

        if np.isscalar(quantiles):
            return out[:, 0]
        return out


__all__ = ["QuantileRegressionForest"]
