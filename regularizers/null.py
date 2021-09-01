class NullRegularization:
    def __init__(self): pass

    def value(self, w):
        return 0

    # Derivative w.r.t. w
    def deriv(self, w):
        return 0