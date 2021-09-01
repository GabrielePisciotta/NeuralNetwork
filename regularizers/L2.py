import numpy as np


class L2Regularization:
    def __init__(self): pass

    def value(self, w):
        return np.sum(w**2)

    # Derivative w.r.t. w
    def deriv(self, w):
        return 2*w