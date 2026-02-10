import numpy as np
from hw2.interfaces import AbstractOptimizer


class VanillaGradientDescent:
    pass
class StochasticGradientDescent:
    pass
class SAGDescent:
    pass
class MomentumDescent:
    pass
class Adam:
    pass

class AnalyticSolutionOptimizer(AbstractOptimizer):
    def optimize(self, model) -> np.ndarray:
        X = model.X
        y = model.y
        w = np.linalg.inv(X.T @ X) @ (X.T @ y)
        model.w = w
        return w

