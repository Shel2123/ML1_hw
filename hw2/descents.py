import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        lrk = self.lambda_ * ((self.s0 / (self.s0 + iteration)) ** self.p)
        return lrk


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """

    def __init__(self,
                 lr_schedule: LearningRateSchedule = TimeDecayLR(),
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                 ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1

        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        self.model.loss_history.append(self.model.compute_loss())
        for epoch in range(self.max_iter):
            d = self._step()
            loss = self.model.compute_loss()
            self.model.loss_history.append(loss)
            if np.isnan(d).any():
                print("Nan in delta")
                break
            if np.sum(np.square(d)) < self.tolerance:
                break


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        cur_lr = self.lr_schedule.get_lr(self.iteration)
        cur_grad = self.model.compute_gradients(X_train, y_train)
        grad_step = -cur_lr * cur_grad
        self.model.w = self.model.w + grad_step
        return grad_step


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        n = self.model.X_train.shape[0]
        batch_idx = np.random.choice(n, size=self.batch_size, replace=False)

        X_batch = self.model.X_train[batch_idx]
        y_batch = self.model.y_train[batch_idx]
        batch_grad = self.model.compute_gradients(X_batch, y_batch)

        cur_lr = self.lr_schedule.get_lr(self.iteration)
        grad_step = -cur_lr * batch_grad

        self.model.w = self.model.w + grad_step
        return grad_step


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects,) + self.model.w.shape, dtype=float)
            self.grad_sum = np.zeros_like(self.model.w, dtype=float)

        batch_idx = np.random.choice(num_objects, size=self.batch_size, replace=False)

        for j in batch_idx:
            g_old = self.grad_memory[j].copy()
            g_new = self.model.compute_gradients(X_train[j:j + 1], y_train[j:j + 1])

            self.grad_memory[j] = g_new
            self.grad_sum += (g_new - g_old) / num_objects

        cur_lr = self.lr_schedule.get_lr(self.iteration)
        grad_step = -cur_lr * self.grad_sum
        self.model.w += grad_step
        return grad_step


class MomentumDescent(BaseDescent):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)
        cur_lr = self.lr_schedule.get_lr(self.iteration)
        cur_grad = self.model.compute_gradients(self.model.X_train, self.model.y_train)

        self.velocity = self.beta * self.velocity + cur_lr * cur_grad
        self.model.w = self.model.w - self.velocity
        return -self.velocity


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        X = self.model.X_train
        y = self.model.y_train
        if self.m is None or self.v is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)

        cur_grad = self.model.compute_gradients(X, y)
        cur_lr = self.lr_schedule.get_lr(self.iteration)

        self.m = self.beta1 * self.m + (1 - self.beta1) * cur_grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(cur_grad)

        t = self.iteration + 1
        m_hat = self.m / (1 - self.beta1 ** t)
        v_hat = self.v / (1 - self.beta2 ** t)
        grad_step = - cur_lr / (np.sqrt(v_hat) + self.eps) * m_hat
        self.model.w += grad_step
        return grad_step


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений
    """
    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        X = self.model.X_train
        y = self.model.y_train

        self.model.w = self.model.loss_function.analytic_solution(X, y)
