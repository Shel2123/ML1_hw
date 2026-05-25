from collections import defaultdict
from typing import Iterable, Any

import numpy as np
from numpy import complexfloating, dtype, floating, ndarray, number, timedelta64
from numpy._typing import _64Bit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin


class TargetEncoder:
    def __init__(
        self,
        cat_features: Iterable[int] | None = None,
        random_state: int | None = None,
        ordered: bool = True,
    ):
        self.cat_features = list(cat_features) if cat_features is not None else []
        self.random_state = random_state
        self.ordered = ordered
        self.maps_: dict[int, dict] = {}
        self.default_: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TargetEncoder":
        y = np.asarray(y)
        classes = np.unique(y)
        y_pos = (y == classes[-1]).astype(np.float64)
        self.default_ = float(y_pos.mean())

        self.maps_ = {}
        for j in self.cat_features:
            col = X[:, j]
            uniques, inverse = np.unique(col, return_inverse=True)
            counts = np.bincount(inverse).astype(np.float64)
            sums = np.bincount(inverse, weights=y_pos)
            means = sums / counts
            self.maps_[j] = dict(zip(uniques.tolist(), means.tolist()))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd

        n_rows, n_cols = X.shape
        out = np.empty((n_rows, n_cols), dtype=np.float64)
        for j in range(n_cols):
            col = X[:, j]
            if j in self.maps_:
                out[:, j] = (
                    pd.Series(col)
                    .map(self.maps_[j])
                    .fillna(self.default_)
                    .to_numpy(dtype=np.float64)
                )
            else:
                out[:, j] = np.asarray(col, dtype=np.float64)
        return out

    def _ordered_encode_column(self, col: np.ndarray, y_pos: np.ndarray) -> np.ndarray:
        n = len(col)
        uniques, inverse = np.unique(col, return_inverse=True)
        n_cats = len(uniques)

        order = np.argsort(inverse, kind='stable')
        sorted_y = y_pos[order]
        sorted_cat = inverse[order]

        group_starts = np.searchsorted(sorted_cat, np.arange(n_cats))
        within_group_pos = np.arange(n) - group_starts[sorted_cat]

        cumsum_global = np.cumsum(sorted_y)
        prefix_offsets = np.zeros(n_cats)
        prefix_offsets[1:] = cumsum_global[group_starts[1:] - 1]
        cum_pos_inclusive = cumsum_global - prefix_offsets[sorted_cat]
        cum_pos_above = cum_pos_inclusive - sorted_y

        with np.errstate(divide='ignore', invalid='ignore'):
            enc_sorted = np.where(
                within_group_pos > 0,
                cum_pos_above / within_group_pos,
                self.default_,
            )

        enc = np.empty(n, dtype=np.float64)
        enc[order] = enc_sorted
        return enc

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)

        if not self.ordered:
            return self.transform(X)

        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        y_pos = (y_arr == classes[-1]).astype(np.float64)

        n_rows, n_cols = X.shape

        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n_rows)
        inv_perm = np.argsort(perm)

        X_perm = X[perm]
        y_pos_perm = y_pos[perm]

        out = np.empty((n_rows, n_cols), dtype=np.float64)
        for j in range(n_cols):
            col = X_perm[:, j]
            if j in self.maps_:
                out[:, j] = self._ordered_encode_column(col, y_pos_perm)
            else:
                out[:, j] = np.asarray(col, dtype=np.float64)
        return out[inv_perm]


class Quantizer:
    def __init__(self, quantization_type: str | None = 'quantile', nbins: int = 255):
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.thresholds_: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Quantizer":
        from sklearn.tree import DecisionTreeRegressor

        self.thresholds_ = []
        for j in range(X.shape[1]):
            col = np.asarray(X[:, j], dtype=np.float64)
            qt = self.quantization_type

            if qt == 'uniform':
                lo, hi = float(col.min()), float(col.max())
                t = np.linspace(lo, hi, self.nbins + 1)[1:-1] if hi > lo else np.array([])

            elif qt == 'quantile':
                qs = np.linspace(0.0, 1.0, self.nbins + 1)[1:-1]
                t = np.unique(np.quantile(col, qs))

            elif qt == 'min_entropy':
                vals, counts = np.unique(col, return_counts=True)
                if len(vals) <= 1:
                    t = np.array([])
                else:
                    cum = np.cumsum(counts)
                    targets = np.linspace(0, cum[-1], self.nbins + 1)[1:-1]
                    idx = np.clip(np.searchsorted(cum, targets), 0, len(vals) - 2)
                    t = np.unique((vals[idx] + vals[idx + 1]) / 2.0)

            elif qt == 'piecewise':
                if y is None:
                    raise ValueError("quantization_type='piecewise' требует y в fit")
                tree = DecisionTreeRegressor(max_leaf_nodes=self.nbins, random_state=0)
                tree.fit(col.reshape(-1, 1), y)
                thr = tree.tree_.threshold
                feat = tree.tree_.feature
                t = np.unique(np.sort(thr[feat >= 0]))

            else:
                raise ValueError(f"unknown quantization_type: {qt}")

            self.thresholds_.append(t.astype(np.float64))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape, dtype=np.float64)
        for j in range(X.shape[1]):
            t = self.thresholds_[j]
            if len(t) == 0:
                out[:, j] = 0.0
            else:
                out[:, j] = np.searchsorted(t, X[:, j], side='right').astype(np.float64)
        return out


def find_best_split(
    feature_vector: np.ndarray,
    grad_vector: np.ndarray,
    hess_vector: np.ndarray,
    l2: float = 1.0,
    min_samples_leaf: int = 1,
) -> tuple[ndarray[Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[floating[Any]]] | ndarray[
    Any, dtype[complexfloating[Any, Any]]] | ndarray[Any, dtype[number[Any]]] | ndarray[
               Any, dtype[timedelta64]] | float | Any, Any, None, None] | tuple[
         ndarray[Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[floating[Any]]] | ndarray[
             Any, dtype[complexfloating[Any, Any]]] | ndarray[Any, dtype[number[Any]]] | ndarray[
             Any, dtype[timedelta64]] | float | Any, Any, float, float]:
    feature_vector = np.asarray(feature_vector, dtype=np.float64)
    grad_vector = np.asarray(grad_vector, dtype=np.float64)
    hess_vector = np.asarray(hess_vector, dtype=np.float64)
    n = len(feature_vector)

    order = np.argsort(feature_vector, kind='stable')
    x = feature_vector[order]
    g = grad_vector[order]
    h = hess_vector[order]

    G_cum = np.cumsum(g)
    H_cum = np.cumsum(h)
    G_total = G_cum[-1]
    H_total = H_cum[-1]
    base = G_total * G_total / (H_total + l2)

    G_l = G_cum[:-1]
    H_l = H_cum[:-1]
    G_r = G_total - G_l
    H_r = H_total - H_l

    gains = G_l * G_l / (H_l + l2) + G_r * G_r / (H_r + l2) - base
    thresholds = (x[:-1] + x[1:]) / 2.0

    diff = x[:-1] < x[1:]
    sizes = np.arange(1, n)
    valid = diff & (sizes >= min_samples_leaf) & ((n - sizes) >= min_samples_leaf)
    gains_masked = np.where(valid, gains, -np.inf)

    if not np.any(valid):
        return thresholds, gains, None, None

    best_i = int(np.argmax(gains_masked))
    return thresholds, gains, float(thresholds[best_i]), float(gains_masked[best_i])


class XGBoostTree:
    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 1,
        l2: float = 1.0,
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.random_state = random_state
        self.tree_: tuple | None = None

    def _leaf_value(self, g_node: np.ndarray, h_node: np.ndarray) -> float:
        return float(g_node.sum() / (h_node.sum() + self.l2))

    def fit(self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray | None = None) -> "XGBoostTree":
        X = np.asarray(X, dtype=np.float64)
        grad = np.asarray(grad, dtype=np.float64)
        hess = np.ones_like(grad) if hess is None else np.asarray(hess, dtype=np.float64)
        self.tree_ = self._build(X, grad, hess, np.arange(len(X)), depth=0)
        return self

    def _build(self, X, grad, hess, idx, depth):
        g_node = grad[idx]
        h_node = hess[idx]
        leaf_value = self._leaf_value(g_node, h_node)

        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return 'leaf', leaf_value

        best_gain = 0.0
        best_f = -1
        best_t = None
        for f in range(X.shape[1]):
            _, _, t, gain = find_best_split(
                X[idx, f], g_node, h_node,
                l2=self.l2, min_samples_leaf=self.min_samples_leaf,
            )
            if t is None or gain is None:
                continue
            if gain > best_gain:
                best_gain = gain
                best_f = f
                best_t = t

        if best_f == -1:
            return 'leaf', leaf_value

        col = X[idx, best_f]
        left_mask = col <= best_t
        idx_l = idx[left_mask]
        idx_r = idx[~left_mask]
        left = self._build(X, grad, hess, idx_l, depth + 1)
        right = self._build(X, grad, hess, idx_r, depth + 1)
        return 'node', best_f, best_t, left, right

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)
        self._predict_into(X, np.arange(len(X)), self.tree_, out)
        return out

    def _predict_into(self, X, idx, node, out):
        if node[0] == 'leaf':
            out[idx] = node[1]
            return
        _, f, t, left, right = node
        col = X[idx, f]
        mask = col <= t
        if mask.any():
            self._predict_into(X, idx[mask], left, out)
        if (~mask).any():
            self._predict_into(X, idx[~mask], right, out)


class Boosting(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
        cat_features: Iterable[int] | None = None,
        l2: float = 0.0,
        ordered_encoding: bool = True,
        subsample: float = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        bagging_temperature: float = 1.0,
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
        quantization_type: str | None = None,
        nbins: int = 255,
        dart: bool = False,
        dropout_rate: float = 0.05,
        loss: str = 'BCE',
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)
        self.feature_indices: list[np.ndarray | None] = [None] * n_estimators

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        self.cat_features = cat_features
        self.ordered_encoding = ordered_encoding
        self._encoder: TargetEncoder | None = None

        self.l2 = l2

        self.subsample = subsample
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature
        self.rsm = rsm
        self.goss = goss
        self.goss_k = goss_k

        self.quantization_type = quantization_type
        self.nbins = nbins
        self._quantizer: Quantizer | None = None

        self.dart = dart
        self.dropout_rate = dropout_rate
        self.tree_weights: list[float] = [1.0] * n_estimators
        self._train_tree_preds: list[np.ndarray] = []
        self._valid_tree_preds: list[np.ndarray] = []

        self.loss = loss
        self.focal_gamma = focal_gamma

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self._init_loss_fns()

    def _init_loss_fns(self) -> None:
        if self.loss == 'BCE':
            self.loss_fn = lambda y, z: np.logaddexp(0.0, -y * z).mean()
            self.grad_fn = lambda y, z: -y * self.sigmoid(-y * z)
            self.hess_fn = lambda y, z: self.sigmoid(y * z) * (1.0 - self.sigmoid(y * z))
        elif self.loss == 'Focal':
            g = self.focal_gamma

            def _focal_loss(y, z):
                p = self.sigmoid(y * z)
                p = np.clip(p, 1e-8, 1.0 - 1e-8)
                return ((1.0 - p) ** g * (-np.log(p))).mean()

            def _focal_grad(y, z):
                p = self.sigmoid(y * z)
                p = np.clip(p, 1e-8, 1.0 - 1e-8)
                one_m_p = 1.0 - p
                return -y * one_m_p ** g * (1.0 - g * p * np.log(p) / one_m_p)

            def _focal_hess(y, z):
                p = self.sigmoid(y * z)
                p = np.clip(p, 1e-8, 1.0 - 1e-8)
                one_m_p = 1.0 - p
                term = (1.0 - g * p * np.log(p) / one_m_p)
                d_term_dp = -g * (np.log(p) + 1.0) / one_m_p - g * p * np.log(p) / one_m_p ** 2
                dgrad_dp = -g * one_m_p ** (g - 1) * (-1.0) * term + one_m_p ** g * d_term_dp
                return np.clip(np.abs(dgrad_dp) * p * one_m_p, 1e-8, None)

            self.loss_fn = _focal_loss
            self.grad_fn = _focal_grad
            self.hess_fn = _focal_hess
        else:
            raise ValueError(f"unknown loss: {self.loss!r}")

    def _to_pm1(self, y: np.ndarray) -> np.ndarray:
        return np.where(y == self.classes_[1], 1.0, -1.0)

    def _make_base_model(self):
        if self.l2 > 0:
            valid = {'max_depth', 'min_samples_leaf', 'random_state'}
            params = {k: v for k, v in self.base_model_params.items() if k in valid}
            params['l2'] = self.l2
            if self.random_state is not None and 'random_state' not in params:
                params['random_state'] = self.random_state + self._step
            return XGBoostTree(**params)
        params = dict(self.base_model_params)
        if self.random_state is not None and 'random_state' not in params:
            params['random_state'] = self.random_state + self._step
        return self.base_model_class(**params)

    def _sample_rng(self, salt: int = 0) -> np.random.Generator:
        seed = None if self.random_state is None else self.random_state + self._step + salt
        return np.random.default_rng(seed)

    def _sample_objects(self, anti_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        n = len(anti_grad)

        if self.goss:
            abs_g = np.abs(anti_grad)
            k = max(1, int(round(n * self.goss_k)))
            top_idx = np.argpartition(-abs_g, k - 1)[:k]
            other_mask = np.ones(n, dtype=bool)
            other_mask[top_idx] = False
            other_idx = np.where(other_mask)[0]
            n_small = max(1, int(round(len(other_idx) * self.subsample)))
            rng = self._sample_rng(salt=1)
            sample_other = rng.choice(other_idx, size=n_small, replace=False)
            amp = (1.0 - self.goss_k) / max(self.subsample, 1e-8)
            weights = np.empty(n, dtype=np.float64)
            weights[top_idx] = 1.0
            weights[sample_other] = amp
            sample_idx = np.concatenate([top_idx, sample_other])
            return sample_idx, weights[sample_idx]

        if self.bootstrap_type == 'Bayesian':
            rng = self._sample_rng(salt=2)
            u = rng.random(n)
            w = (-np.log(np.clip(u, 1e-12, 1.0))) ** self.bagging_temperature
            return np.arange(n), w

        if self.bootstrap_type == 'Bernoulli' and self.subsample < 1.0:
            rng = self._sample_rng(salt=3)
            mask = rng.random(n) < self.subsample
            if not mask.any():
                mask[rng.integers(n)] = True
            return np.where(mask)[0], None

        return np.arange(n), None

    def _sample_features(self, n_features: int) -> np.ndarray:
        if self.rsm >= 1.0:
            return np.arange(n_features)
        n_pick = max(1, int(round(n_features * self.rsm)))
        rng = self._sample_rng(salt=7)
        return np.sort(rng.choice(n_features, size=n_pick, replace=False))

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y_pm = self._to_pm1(y)
        z = self._train_predictions

        if self.dart and self._step > 0:
            rng = self._sample_rng(salt=11)
            n_trees = self._step
            drop_mask = rng.random(n_trees) < self.dropout_rate
            if not drop_mask.any():
                drop_mask[rng.integers(n_trees)] = True
            dropped_idx = np.where(drop_mask)[0]
            k = len(dropped_idx)
            scale = 1.0 / (k + 1)
            z_dropped = np.zeros_like(z)
            for j in dropped_idx:
                z_dropped += (
                    self.learning_rate * self.gammas[j] * self.tree_weights[j]
                    * self._train_tree_preds[j]
                )
            z_fit = z - z_dropped
        else:
            dropped_idx = np.array([], dtype=int)
            scale = 1.0
            z_fit = z

        anti_grad = -self.grad_fn(y_pm, z_fit)
        sample_idx, sample_w = self._sample_objects(anti_grad)
        feat_idx = self._sample_features(X.shape[1])

        X_sub = X[sample_idx][:, feat_idx]
        g_sub = anti_grad[sample_idx]
        model = self._make_base_model()

        if self.l2 > 0:
            hess_full = np.clip(self.hess_fn(y_pm, z_fit), 1e-8, None)
            h_sub = hess_full[sample_idx]
            if sample_w is not None:
                g_sub = g_sub * sample_w
                h_sub = h_sub * sample_w
            model.fit(X_sub, g_sub, h_sub)
            new_predictions = model.predict(X[:, feat_idx])
            gamma = 1.0
        else:
            if sample_w is not None:
                model.fit(X_sub, g_sub, sample_weight=sample_w)
            else:
                model.fit(X_sub, g_sub)
            new_predictions = model.predict(X[:, feat_idx])
            gamma = self._find_optimal_gamma(y_pm, z_fit, new_predictions)

        self.models[self._step] = model
        self.gammas[self._step] = gamma
        self.feature_indices[self._step] = feat_idx
        self.tree_weights[self._step] = scale

        if self.dart:
            self._train_tree_preds.append(new_predictions)
            weight_change: dict[int, tuple[float, float]] = {}
            for j in dropped_idx:
                w_old = self.tree_weights[j]
                w_new = w_old * (k / (k + 1))
                z = z + (
                    self.learning_rate * self.gammas[j]
                    * (w_new - w_old) * self._train_tree_preds[j]
                )
                self.tree_weights[j] = w_new
                weight_change[int(j)] = (w_old, w_new)
            z = z + self.learning_rate * gamma * scale * new_predictions
            self._last_dart_dropped = list(dropped_idx)
            self._last_dart_weight_change = weight_change
        else:
            z = z + self.learning_rate * gamma * new_predictions
        self._train_predictions = z

        self.history["train_loss"].append(self.loss_fn(y_pm, z))
        self.history["train_roc_auc"].append(
            roc_auc_score(y_pm == 1, self.sigmoid(z))
        )

        self._step += 1

    @staticmethod
    def _metric_is_better(current: float, best: float, metric_name: str) -> bool:
        if 'loss' in metric_name.lower():
            return current < best
        return current > best

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        use_best_model: bool = False,
    ) -> None:

        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки

        if self.cat_features is not None:
            self._encoder = TargetEncoder(
                self.cat_features,
                random_state=self.random_state,
                ordered=self.ordered_encoding,
            )
            X_train = self._encoder.fit_transform(X_train, y_train)
            if eval_set is not None:
                X_valid_raw, y_valid = eval_set
                eval_set = (self._encoder.transform(X_valid_raw), y_valid)

        if self.quantization_type is not None:
            self._quantizer = Quantizer(
                quantization_type=self.quantization_type, nbins=self.nbins
            )
            self._quantizer.fit(X_train, y_train if self.quantization_type == 'piecewise' else None)
            X_train = self._quantizer.transform(X_train)
            if eval_set is not None:
                X_valid_q, y_valid = eval_set
                eval_set = (self._quantizer.transform(X_valid_q), y_valid)

        self._train_predictions = np.zeros(X_train.shape[0])
        self._step = 0
        self.n_features_in_ = X_train.shape[1]
        self.tree_weights = [1.0] * self.n_estimators
        self._train_tree_preds = []
        self._valid_tree_preds = []
        self._last_dart_dropped = []
        self._last_dart_weight_change = {}

        has_eval = eval_set is not None
        if has_eval:
            X_valid, y_valid = eval_set
            y_pm_valid = self._to_pm1(y_valid)
            val_predictions = np.zeros(X_valid.shape[0])

        es_rounds = self.early_stopping_rounds or 0
        eval_metric = self.eval_metric
        if es_rounds and eval_metric is None:
            eval_metric = 'val_loss' if has_eval else 'train_loss'

        best_metric: float | None = None
        best_step_idx = 0
        rounds_no_improve = 0

        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        for _ in estimator_range:
            self.partial_fit(X_train, y_train)

            if has_eval:
                last_model = self.models[self._step - 1]
                last_gamma = self.gammas[self._step - 1]
                last_feat = self.feature_indices[self._step - 1]
                last_w = self.tree_weights[self._step - 1]
                last_val_pred = last_model.predict(X_valid[:, last_feat])
                if self.dart:
                    self._valid_tree_preds.append(last_val_pred)
                    for j, (old_w, new_w) in self._last_dart_weight_change.items():
                        val_predictions = val_predictions + (
                            self.learning_rate * self.gammas[j]
                            * (new_w - old_w) * self._valid_tree_preds[j]
                        )
                    val_predictions = val_predictions + (
                        self.learning_rate * last_gamma * last_w * last_val_pred
                    )
                else:
                    val_predictions = (
                        val_predictions
                        + self.learning_rate * last_gamma * last_val_pred
                    )
                self.history['val_loss'].append(self.loss_fn(y_pm_valid, val_predictions))
                self.history['val_roc_auc'].append(
                    roc_auc_score(y_pm_valid == 1, self.sigmoid(val_predictions))
                )

            if es_rounds:
                current = self.history[eval_metric][-1]
                improved = (
                    best_metric is None
                    or self._metric_is_better(current, best_metric, eval_metric)
                )
                if improved:
                    best_metric = current
                    best_step_idx = self._step - 1
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    if rounds_no_improve >= es_rounds:
                        break

        n_done = self._step
        if use_best_model and es_rounds and best_metric is not None:
            keep = best_step_idx + 1
        else:
            keep = n_done
        self.models = self.models[:keep]
        self.gammas = self.gammas[:keep]
        self.feature_indices = self.feature_indices[:keep]
        self.tree_weights = self.tree_weights[:keep]
        if self.dart:
            self._train_tree_preds = self._train_tree_preds[:keep]
            self._valid_tree_preds = self._valid_tree_preds[:keep]

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._encoder is not None:
            X = self._encoder.transform(X)
        if self._quantizer is not None:
            X = self._quantizer.transform(X)
        z = np.zeros(X.shape[0])
        for model, gamma, feat_idx, w in zip(
            self.models, self.gammas, self.feature_indices, self.tree_weights
        ):
            if isinstance(model, int):
                continue
            X_in = X if feat_idx is None else X[:, feat_idx]
            z = z + self.learning_rate * gamma * w * model.predict(X_in)
        proba_pos = self.sigmoid(z)
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray, 
        new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])

    @staticmethod
    def _xgbt_split_counts(node, n_features: int) -> np.ndarray:
        counts = np.zeros(n_features, dtype=np.float64)

        def walk(nd):
            if nd is None or nd[0] == 'leaf':
                return
            _, f, _t, left, right = nd
            counts[f] += 1.0
            walk(left)
            walk(right)

        walk(node)
        return counts

    def get_feature_importance(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        type: str = 'split',
        sample_size: int | None = 5000,
    ) -> np.ndarray:
        if type == 'split':
            return self._feature_importance_split()
        if type == 'gain':
            if X is None or y is None:
                raise ValueError("type='gain' требует X и y")
            return self._feature_importance_gain(X, y, sample_size=sample_size)
        raise NotImplementedError(f"unknown type: {type!r}")

    def _feature_importance_split(self) -> np.ndarray:
        n_feat = self.n_features_in_
        importances = np.zeros(n_feat, dtype=np.float64)

        for model, gamma, feat_idx, w in zip(
            self.models, self.gammas, self.feature_indices, self.tree_weights
        ):
            if isinstance(model, int):
                continue
            if hasattr(model, 'feature_importances_'):
                imp_local = np.asarray(model.feature_importances_, dtype=np.float64)
            elif isinstance(model, XGBoostTree):
                imp_local = self._xgbt_split_counts(model.tree_, len(feat_idx))
                s = imp_local.sum()
                if s > 0:
                    imp_local = imp_local / s
            else:
                continue
            importances[feat_idx] += abs(gamma * w) * imp_local

        total = importances.sum()
        if total > 0:
            importances = importances / total
        return importances

    def _feature_importance_gain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_size: int | None = 5000,
    ) -> np.ndarray:
        from sklearn.tree import DecisionTreeRegressor as _DTR

        X = np.asarray(X)
        if self._encoder is not None:
            X = self._encoder.transform(X)
        if self._quantizer is not None:
            X = self._quantizer.transform(X)
        X = np.asarray(X, dtype=np.float64)

        n = X.shape[0]
        if sample_size is not None and n > sample_size:
            rng = np.random.default_rng(self.random_state)
            sub = rng.choice(n, size=sample_size, replace=False)
            X = X[sub]
            y = np.asarray(y)[sub]
            n = sample_size

        y_pm = self._to_pm1(y)
        z = np.zeros(n)
        for model, gamma, feat_idx, w in zip(
            self.models, self.gammas, self.feature_indices, self.tree_weights
        ):
            if isinstance(model, int):
                continue
            z = z + self.learning_rate * gamma * w * model.predict(X[:, feat_idx])
        anti_grad = -self.grad_fn(y_pm, z)

        n_feat = self.n_features_in_
        importances = np.zeros(n_feat, dtype=np.float64)

        for model, gamma, feat_idx, w in zip(
            self.models, self.gammas, self.feature_indices, self.tree_weights
        ):
            if isinstance(model, int) or not isinstance(model, _DTR):
                continue
            local_contrib = self._gain_walk_sklearn(
                model, X[:, feat_idx], anti_grad,
                lr_gamma=self.learning_rate * gamma * w,
            )
            importances[feat_idx] += local_contrib

        total = importances.sum()
        if total > 0:
            importances = importances / total
        return importances

    @staticmethod
    def _gain_walk_sklearn(model, X_in: np.ndarray, anti_grad: np.ndarray, lr_gamma: float) -> np.ndarray:
        tree = model.tree_
        n_local = X_in.shape[1]
        contrib = np.zeros(n_local, dtype=np.float64)

        values = tree.value.reshape(-1).astype(np.float64)
        features = tree.feature
        thresholds = tree.threshold
        left = tree.children_left
        right = tree.children_right

        n = X_in.shape[0]
        node_ids = np.zeros(n, dtype=np.int64)

        for _ in range(int(tree.max_depth) + 1):
            cur_feat = features[node_ids]
            active = cur_feat >= 0
            if not active.any():
                break
            idx_act = np.where(active)[0]
            cur = node_ids[idx_act]
            f = features[cur]
            t = thresholds[cur]
            go_left = X_in[idx_act, f] <= t
            next_node = np.where(go_left, left[cur], right[cur])

            delta = values[next_node] - values[cur]
            contrib_obj = np.abs(lr_gamma * anti_grad[idx_act] * delta)
            np.add.at(contrib, f, contrib_obj)

            node_ids[idx_act] = next_node

        return contrib

    def plot_history(self, keys: str | Iterable[str]):
        import matplotlib.pyplot as plt

        if isinstance(keys, str):
            keys = [keys]
        else:
            keys = list(keys)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        for key in keys:
            if key not in self.history or len(self.history[key]) == 0:
                continue
            values = np.asarray(self.history[key])
            ax.plot(np.arange(1, len(values) + 1), values, label=key)
        ax.set_xlabel('iteration')
        ax.set_ylabel('metric')
        ax.set_title('Training history')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()
