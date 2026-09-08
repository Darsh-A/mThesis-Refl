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
        """
        leaf_test = self.rf.apply(X)                 # (n_test, n_estimators)
        n_test = leaf_test.shape[0]
        n_train = self.leaf_train_.shape[0]
        K = self.rf.n_estimators

        weights = np.zeros((n_test, n_train))
        for t in range(K):
            lt = self.leaf_train_[:, t]              # train leaf ids, tree t
            le = leaf_test[:, t]                     # test leaf ids, tree t
            match = le[:, None] == lt[None, :]       # (n_test, n_train) bool
            cnt = match.sum(axis=1).astype(float)
            cnt[cnt == 0.0] = 1.0                    # guard (shouldn't happen)
            weights += match / (K * cnt[:, None])

        row_sum = weights.sum(axis=1, keepdims=True)
        weights = weights / np.where(row_sum == 0.0, 1.0, row_sum)

        q = np.atleast_1d(np.asarray(quantiles, dtype=float))
        out = np.empty((n_test, len(q)))
        for j in range(n_test):
            c = np.cumsum(weights[j][self._order])
            idx = np.searchsorted(c, q)
            idx = np.clip(idx, 0, n_train - 1)
            out[j] = self._y_sorted[idx]

        if np.isscalar(quantiles):
            return out[:, 0]
        return out


__all__ = ["QuantileRegressionForest"]
