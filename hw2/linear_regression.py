import numpy as np
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC
from scipy.sparse.linalg import svds
from scipy.sparse import issparse


class MSELoss(LossFunction, LossFunctionClosedFormMixin):
    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        n = y.shape[0]
        r = X @ w - y
        Q = (1.0 / n) * (r.T @ r)
        return float(Q)

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        n = y.shape[0]
        grad = (2.0 / n) * (X.T @ (X @ w - y))
        return grad

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса;
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)

    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        return np.linalg.inv(X.T @ X) @ (X.T @ y)

    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        n, m = X.shape
        kmax = min(n, m) - 1

        U, s, Vt = svds(
            X, k=kmax, which="LM",
            solver="arpack",
            tol=0,
            maxiter=None,
        )
        cut = 1e-8 * max(n, m) * s[0]

        s_inv = np.where(s > cut, 1.0 / s, 0.0)
        w = Vt.T @ (s_inv * (U.T @ y))

        return w

class L2Regularization(LossFunction):
    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0,
                 analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate

        # analytic_solution_func is meant to be passed separately,
        # as it is not linear to core solution

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.gradient(X, y, w)

        penalty_part = ...

        raise NotImplementedError()
        # TODO: implement


class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
            self,
            optimizer: AbstractOptimizer,
            # l2_coef: float = 0.0,
            loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \\hat{y}
        """
        if self.w is None:
            raise RuntimeError("Not fitted: w = None")
        if hasattr(X, "to_numpy") and not issparse(X):
            X = X.to_numpy()
        return X @ self.w

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is not None and y_batch is not None:
            return self.loss_function.gradient(X_batch, y_batch, self.w)
        return self.loss_function.gradient(self.X_train, self.y_train, self.w)

    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is not None and y_batch is not None:
            return self.loss_function.loss(X_batch, y_batch, self.w)
        return self.loss_function.loss(self.X_train, self.y_train, self.w)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.

        X: np.ndarray,
        y: np.ndarray
        """
        self.loss_history = []
        self.optimizer.iteration = 0
        if hasattr(X, "to_numpy") and not issparse(X):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        y = np.asarray(y).reshape(-1)
        self.X_train = X
        self.y_train = y
        self.w = np.zeros(X.shape[1], dtype=float)
        self.optimizer.optimize()
