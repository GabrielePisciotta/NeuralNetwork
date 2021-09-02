import numpy as np
from activationfunctions.sigmoid import Sigmoid
from activationfunctions.linear import  Linear
from activationfunctions.softplus import Softplus
from lossfunctions.MSE import MSE
from weightupdaters.WeightUpdater import CompoundWeightUpdater, LBFGSWeightUpdater
from math import sqrt
class Layer:

    def __init__(self, n_of_neurons, n_of_features, lossfunction, regtype, reglambda = 0, type = 'hidden', momentumAlpha=0, momentumBeta=0, weightsInitializer='default', activationFunction='sigmoid'):
        self.n_of_neurons = n_of_neurons
        self.n_of_features = n_of_features
        self.weightsInitializer = weightsInitializer
        self.bias    = np.zeros((1, n_of_neurons))
        self.type = type

        if activationFunction == 'sigmoid':
            self.activation_function = Sigmoid()
        elif activationFunction == 'softplus':
            self.activation_function = Softplus()
        elif activationFunction == 'linear':
            self.activation_function = Linear()


        self.loss_function = lossfunction
        
        self.weights_updater = CompoundWeightUpdater(
            momentumType='default', regularizationType=regtype, 
            alpha=momentumAlpha, beta=momentumBeta, 
            lamb=reglambda
            )
        
        #print("{} layer created with {} neurons!".format(self.type, self.n_of_neurons))
        
        self.initializeWeights()

    def getWeightsTemp(self):
        return self.weights[0][0]

    def evaluate_input(self, input):
        self.input = input
        self.net = (input @ self.weights) + self.bias
        self.output = self.activation_function.evaluate(self.net)
        return self.output

    # @param value: since this function is called from NeuralNetwork class, we pass the label as `value` for the
    #               output layer, and we pass delta.dot(self.weights.T) for the hidden layer.
    # @return the gradient
    def backward(self, value):
        if self.type == 'output':
            loss_deriv_by_o = (-1) * self.loss_function.deriv(value, self.output)
            activ_deriv = self.activation_function.deriv(self.net)
            self.delta = loss_deriv_by_o * activ_deriv
        else:
            self.delta = value * self.activation_function.deriv(self.net)

        return self.delta


    def initializeWeights(self):
        if self.weightsInitializer=='default':
            self.weights = np.random.uniform(-0.2 , 0.2, size=(self.n_of_features, self.n_of_neurons))
            self.bias    = np.zeros((1, self.n_of_neurons))

        # As stated in Glorot and Bengio (2010)
        elif self.weightsInitializer == 'xavier':
            if self.activation_function == 'sigmoid':
                k = 4
            else:
                k = 1
            if self.type == 'hidden':
                xavier = k * sqrt(6)/( sqrt( self.n_of_features + self.n_of_neurons) )
                self.weights = np.random.uniform(-xavier, +xavier, size=(self.n_of_features, self.n_of_neurons))
            else:
                self.weights = np.random.uniform(-1, 1, size=(self.n_of_features, self.n_of_neurons))
            self.bias    = np.zeros((1, self.n_of_neurons))

        elif self.weightsInitializer == 'larger':
            self.weights = np.random.uniform(-5 , 5, size=(self.n_of_features, self.n_of_neurons))
            self.bias    = np.zeros((1, self.n_of_neurons))

        self.weights_updater.reset()


class LBFGSLayer:

    def __init__(self, n_of_neurons, n_of_features, lossfunction, regtype, reglambda=0, type='hidden', momentumAlpha=0,
                 momentumBeta=0, weightsInitializer='default', activationFunction='sigmoid'):
        self.n_of_neurons = n_of_neurons
        self.n_of_features = n_of_features
        self.weightsInitializer = weightsInitializer
        self.bias = np.zeros((1, n_of_neurons))
        self.accumulated_gradient = np.zeros((n_of_features, n_of_neurons))
        self.delta = np.zeros((n_of_features+1, n_of_neurons))
        self.old_delta = np.zeros((n_of_features+1, n_of_neurons))
        self.deltaweights = np.zeros((n_of_features, n_of_neurons))
        self.direction = np.zeros((n_of_features, n_of_neurons))
        self.k = 0

        # Curvature informations
        self.past_curvatures = []
        self.type = type

        if activationFunction == 'sigmoid':
            self.activation_function = Sigmoid()
        elif activationFunction == 'softplus':
            self.activation_function = Softplus()
        elif activationFunction == 'linear':
            self.activation_function = Linear()

        self.loss_function = lossfunction

        self.weights_updater = LBFGSWeightUpdater(
            momentumType='default', regularizationType=regtype,
            alpha=momentumAlpha, beta=momentumBeta,
            lamb=reglambda
        )

        # print("{} layer created with {} neurons!".format(self.type, self.n_of_neurons))

        self.initializeWeights()

    def getWeightsTemp(self):
        return self.weights[0][0]

    def computeGradientWeight(self):
        self.deltaweights = self.input.T @ self.delta

    def GetDirection(self):
        return self.direction


    def evaluate_input(self, input, in_linesearch=False):
        if not in_linesearch:
            self.input = input

        # Update the net
        net = (input @ self.weights) + self.bias
        output = self.activation_function.evaluate(net)

        if not in_linesearch:
            self.net = net
            self.output = output

        return output

    # @param value: since this function is called from NeuralNetwork class, we pass the label as `value` for the
    #               output layer, and we pass delta.dot(self.weights.T) for the hidden layer.
    # @return the gradient
    def backward(self, value, in_linesearch=False):
        if self.type == 'output':
            loss_deriv_by_o = (-1) * self.loss_function.deriv(value, self.output)
            activ_deriv = self.activation_function.deriv(self.net)
            delta = loss_deriv_by_o * activ_deriv
        else:
            delta = value * self.activation_function.deriv(self.net)

        if not in_linesearch:
            self.delta = delta

        return delta

    def initializeWeights(self):
        if self.weightsInitializer == 'default':
            self.weights = np.random.uniform(-0.2, 0.2, size=(self.n_of_features, self.n_of_neurons))
            self.bias = np.ones((1, self.n_of_neurons))

        # As stated in Glorot and Bengio (2010)
        elif self.weightsInitializer == 'xavier':
            if self.activation_function == 'sigmoid':
                k = 4
            else:
                k = 1
            if self.type == 'hidden':
                xavier = k * sqrt(6) / (sqrt(self.n_of_features + self.n_of_neurons))
                self.weights = np.random.uniform(-xavier, +xavier, size=(self.n_of_features, self.n_of_neurons))
            else:
                self.weights = np.random.uniform(-1, 1, size=(self.n_of_features, self.n_of_neurons))
            self.bias = np.zeros((1, self.n_of_neurons))

        elif self.weightsInitializer == 'larger':
            self.weights = np.random.uniform(-5, 5, size=(self.n_of_features, self.n_of_neurons))
            self.bias = np.zeros((1, self.n_of_neurons))

        self.weights_updater.reset()

