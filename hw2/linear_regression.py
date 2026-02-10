from hw2.descents import AnalyticSolutionOptimizer
from hw2.interfaces import LossFunction, LinearRegressionInterface
import numpy as np

class CustomLinearRegression(LinearRegressionInterface):
    def __init__(self, optimizer: AnalyticSolutionOptimizer, loss_function: LossFunction):
        self.optimizer = optimizer
        self.loss_func = loss_function

        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.w: np.ndarray | None = None

    @property
    def X(self) -> np.ndarray:
        if self.X_train is None:
            raise RuntimeError("Not fitted: X = None")
        return self.X_train

    @property
    def y(self) -> np.ndarray:
        if self.y_train is None:
            raise RuntimeError("Not fitted: y = None")
        return self.y_train

    @staticmethod
    def _prepare_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X

    @staticmethod
    def _prepare_y(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return y

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = self._prepare_X(X)
        self.y_train = y
        self.optimizer.optimize(self)
        return self

    def predict(self, X: np.ndarray):
        if self.w is None:
            raise RuntimeError("Not fitted: w = None")
        X_ = self._prepare_X(X)
        return X_ @ self.w


class MSELoss(LossFunction):
    @staticmethod
    def _check_batch(X: np.ndarray, y: np.ndarray, indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if indices is not None:
            return X[indices], y[indices]
        return X, y


    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, indices: np.ndarray | None = None) -> float:
        X, y = self._check_batch(X, y, indices)
        n = y.shape[0]
        r = X @ w - y
        Q = (1.0 / n) * (r.T @ r)
        print(f"y.shape = {y.shape}")
        return float(Q)


    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        X, y = self._check_batch(X, y, indices)
        n = y.shape[0]
        grad = (2.0 / n) * (X.T @ (X @ w - y))
        print(f"y.shape = {y.shape}")
        return grad


    @staticmethod
    def _plain_analytic_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(X.T @ X) @ (X.T @ y)