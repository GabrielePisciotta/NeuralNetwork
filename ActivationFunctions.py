import numpy as np

class Sigmoid:
    def evaluate(self, x):
        return (1 / (1 + np.exp(-x)))
    def deriv(self, x):
        return (1 - self.evaluate(x)) * self.evaluate(x)

class Linear:
    def evaluate(self, x):
        return x
    def deriv(self, x):
        return np.ones(x.shape[1])

class Softplus:
    def evaluate(self, x):
        return np.log(1 + np.exp(x))

    def deriv(self, x):
        return (1 / (1 + np.exp(-x)))
