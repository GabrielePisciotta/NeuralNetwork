import numpy as np

class SquareLoss:
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.sum((y - y_pred)**2)

    # Derivative w.r.t. y_pred
    def deriv(self, y, y_pred):
        return -(y - y_pred)