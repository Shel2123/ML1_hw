import numpy as np
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC
from scipy.sparse.linalg import svds
import cupy as cp
import cupyx


class MSELoss(LossFunction, LossFunctionClosedFormMixin):
    def __init__(self, analytic_solution_func: Callable[[cp.ndarray, cp.ndarray], cp.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

    def loss(self, X: cp.ndarray, y: cp.ndarray, w: cp.ndarray) -> float:
        """
        X: cp.ndarray, матрица регрессоров
        y: cp.ndarray, вектор таргета
        w: cp.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        n = y.shape[0]
        r = X @ w - y
        Q = cp.dot(r, r) / n
        return float(Q.item())

    def gradient(self, X: cp.ndarray, y: cp.ndarray, w: cp.ndarray) -> cp.ndarray:
        """
        X: cp.ndarray, матрица регрессоров
        y: cp.ndarray, вектор таргета
        w: cp.ndarray, вектор весов

        returns: cp.ndarray, численный градиент MSE в точке w
        """
        n = y.shape[0]
        grad = (2.0 / n) * (X.T @ (X @ w - y))
        return grad

    def analytic_solution(self, X: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: cp.ndarray, матрица регрессоров
        y: cp.ndarray, вектор таргета

        returns: cp.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса;
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)

    @classmethod
    def _plain_analytic_solution(cls, X: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        """
        X: cp.ndarray, матрица регрессоров
        y: cp.ndarray, вектор таргета

        returns: cp.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        return cp.linalg.solve(X.T @ X, X.T @ y)

    @classmethod
    def _svd_analytic_solution(cls, X: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        """
        X: cp.ndarray, матрица регрессоров
        y: cp.ndarray, вектор таргета

        returns: cp.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        if cupyx.scipy.sparse.isspmatrix(X):
            n, m = X.shape
            kmax = min(n, m) - 1
            U, s, Vt = cupyx.scipy.sparse.linalg.svds(
                X, k=kmax, which="LM", tol=0.0, maxiter=None
            )
            cut = 1e-8 * max(n, m) * s.max()
            s_inv = cp.where(s > cut, 1.0 / s, 0.0)
            w = Vt.T @ (s_inv * (U.T @ y))
            return w

        U, s, Vt = cp.linalg.svd(X, full_matrices=False)
        n, m = X.shape
        cut = 1e-8 * max(n, m) * s.max()
        s_inv = cp.where(s > cut, 1.0 / s, 0.0)
        return (Vt.T * s_inv) @ (U.T @ y)

class L2Regularization(LossFunction):
    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0,
                 analytic_solution_func: Callable[[cp.ndarray, cp.ndarray], cp.ndarray] = None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate

        # analytic_solution_func is meant to be passed separately,
        # as it is not linear to core solution

    def loss(self, X, y, w):
        core = self.core_loss.loss(X, y, w)
        w_reg = w.copy()
        w_reg[-1] = 0.0
        penalty = 0.5 * self.mu_rate * float(cp.sum(w_reg ** 2).item())
        return core + penalty

    def gradient(self, X, y, w):
        core_grad = self.core_loss.gradient(X, y, w)
        w_reg = w.copy()
        w_reg[-1] = 0.0
        return core_grad + self.mu_rate * w_reg


class LogCosh(LossFunction):
    def gradient(self, X: cp.ndarray, y: cp.ndarray, w: cp.ndarray) -> cp.ndarray:
        n = y.shape[0]
        r = X @ w - y
        grad = (1.0 / n) * (X.T @ cp.tanh(r))
        return grad
    def loss(self, X: cp.ndarray, y: cp.ndarray, w: cp.ndarray) -> float:
        n = y.shape[0]
        r = X @ w - y
        Q = (1.0 / n) * cp.sum(cp.log(cp.cosh(r)))
        return float(Q.item())

class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        self.delta = delta
    def gradient(self, X: cp.ndarray, y: cp.ndarray, w: cp.ndarray) -> cp.ndarray:
        n = y.shape[0]
        r = X @ w - y
        abs_r = cp.abs(r)

        g = cp.where(abs_r < self.delta, r, self.delta * cp.sign(r))
        grad = (1.0 / n) * (X.T @ g)
        return grad

    def loss(self, X: cp.ndarray, y: cp.ndarray, w: cp.ndarray) -> float:
        n = y.shape[0]
        r = X @ w - y
        abs_r = cp.abs(r)

        quad = 0.5 * r ** 2
        lin = self.delta * abs_r - 0.5 * self.delta ** 2

        Q = (1.0 / n) * cp.sum(cp.where(abs_r < self.delta, quad, lin))
        return float(Q.item())

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

    def predict(self, X: cp.ndarray) -> cp.ndarray:
        """
        returns: cp.ndarray, вектор \\hat{y}
        """
        if hasattr(X, "to_cupy"):
            X = X.to_cupy()
        X = cp.asarray(X, dtype=cp.float32)
        return X @ self.w

    def compute_gradients(self, X_batch: cp.ndarray | None = None, y_batch: cp.ndarray | None = None) -> cp.ndarray:
        """
        returns: cp.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is not None and y_batch is not None:
            return self.loss_function.gradient(X_batch, y_batch, self.w)
        return self.loss_function.gradient(self.X_train, self.y_train, self.w)

    def compute_loss(self, X_batch: cp.ndarray | None = None, y_batch: cp.ndarray | None = None) -> float:
        """
        returns: cp.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is not None and y_batch is not None:
            return self.loss_function.loss(X_batch, y_batch, self.w)
        return self.loss_function.loss(self.X_train, self.y_train, self.w)

    def fit(self, X: cp.ndarray, y: cp.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.

        X: cp.ndarray,
        y: cp.ndarray
        """
        self.loss_history = []
        self.optimizer.iteration = 0
        if hasattr(X, "to_cupy"):
            X = X.to_cupy()
        if hasattr(y, "to_cupy"):
            y = y.to_cupy()
        self.X_train = X
        self.y_train = y
        self.w = cp.zeros(X.shape[1], dtype=cp.float32)
        self.optimizer.optimize()
