from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    @abstractmethod
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, indices: np.ndarray | None = None) -> float:
        pass

    @abstractmethod
    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        pass

class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass

class LinearRegressionInterface(ABC):
    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass

class AbstractOptimizer(ABC):
    @abstractmethod
    def optimize(self, model: LinearRegressionInterface) -> np.ndarray:
        pass