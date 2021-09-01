import numpy as np

class Linear:
    def evaluate(self, x):
        return x
    def deriv(self, x):
        return np.ones(x.shape[1])