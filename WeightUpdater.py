import numpy as np

from RegularizationFunctions import NullRegularization, L2Regularization

class WeightUpdater(object):
    def __init__(self):
        self.delta = 0.0

    def update(self, weights, bias, input, delta, learning_rate):
        return NotImplementedError()

class BaseMomentum():
    def __init__(self, beta):
        self.beta = beta # Moving average coefficient, 0 <= beta <= 1
        # Set beta = 0 to avoid moving average

        self.delta_weights = 0
        self.delta_bias = 0
        
    def value(self):
        return self.delta_weights, self.delta_bias
    
    def update(self, delta_weights, delta_bias):
        self.delta_weights = (self.beta * self.delta_weights) + ((1 - self.beta) * delta_weights)
        self.delta_bias = (self.beta * self.delta_bias) + ((1 - self.beta) * delta_bias)
        
    def reset(self):
        self.delta_weights = 0
        self.delta_bias = 0
        
class NullMomentum():
    def __init__(self, beta): pass

    def value(self): return 0, 0
    
    def update(self, delta_weights, delta_bias): pass

    def reset(self): pass
    

class CompoundWeightUpdater(WeightUpdater):
    def __init__(self, momentumType, regularizationType, alpha, beta, lamb):
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        
        if (alpha == 0): momentumType = 'none'
        if (lamb == 0): regularizationType = 'none'
        
        if momentumType == 'none':
            self.Momentum = NullMomentum(beta)
        elif momentumType == 'default':
            self.Momentum = BaseMomentum(beta)
        else:
            assert(False), \
                "Invalid MOMENTUM"
                
        if regularizationType == 'none':
            self.Regularization = NullRegularization() # TODO: insert placeholder
        elif regularizationType == 'l2':
            self.Regularization = L2Regularization()
        else:
            assert(False), \
                "Invalid REGULARIZATION"
            
    def update(self, weights, bias, input, delta, learning_rate):
        # Compute the new delta component
        delta_weights = input.T @ delta
        delta_bias = np.sum(delta, axis=0, keepdims=True)
        
        # Compute the momentum term
        MomentumWeights, MomentumBias = self.Momentum.value()
        
        # Compute the regularization term
        RegularizationWeights, RegularizationBias = self.Regularization.deriv(weights), 0
        
        # Compute the overall delta term
        delta_weights = \
            ( learning_rate * delta_weights) + \
            ( self.alpha * MomentumWeights ) - \
            ( self.lamb * RegularizationWeights )
            
        delta_bias = \
            ( learning_rate * delta_bias) + \
            ( self.alpha * MomentumBias ) - \
            ( self.lamb * RegularizationBias )
        
        # Update the values
        weights += delta_weights
        bias += delta_bias
        
        # Update the momentum state
        self.Momentum.update(delta_weights, delta_bias)
        
        return weights, bias
        
    def reset(self):
        self.Momentum.reset()