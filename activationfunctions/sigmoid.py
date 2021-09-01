import numpy as np

class Sigmoid:
    def evaluate(self, x):
        return (1 / (1 + np.exp(-x)))
        #return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))
    def deriv(self, x):
        return (1 - self.evaluate(x)) * self.evaluate(x)