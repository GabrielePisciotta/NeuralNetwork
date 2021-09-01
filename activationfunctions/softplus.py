import numpy as np

class Softplus:
    def evaluate(self, x):
        return np.log(1 + np.exp(x))
    def deriv(self, x):
        return (1 / (1 + np.exp(-x)))