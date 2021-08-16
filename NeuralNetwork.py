import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from Layer import Layer, LBFGSLayer
from LossFunctions import SquareLoss
from TrainingAlgorithms import MiniBatchLearning, LBFGSTraining
from Utilities import plot_error_curve, plot_accuracy_mee
from Metric import Accuracy, MEE

class NeuralNetwork:

    def __init__(self, training_set, training_labels, losstype, regtype='none', reglambda=0, learning_rate=0.1, epochs=1000, algo='minibatch', batchSize=-1, momentumAlpha=0, momentumBeta=0, restart=1, numHiddenLayers=1, numOfUnitsPerLayer=1, numOfUnitPerOutput=-1, weightsInitializer='default', task='classification', activationFunction='sigmoid', kind='SGD'):
        assert momentumBeta >= 0 and momentumBeta <= 1, \
            "Invalid MOMENTUM BETA"

        self.kind = kind

        self.learning_rate = learning_rate
        self.momentumAlpha = momentumAlpha
        self.momentumBeta = momentumBeta
        
        if losstype == 'squareloss':
            self.loss_function = SquareLoss()
        else:
            assert(False), \
                "Invalid LOSS function"
                
        self.regularization_type = regtype
        self.regularization_lambda = reglambda
        
        self.training_set = training_set
        self.labels = training_labels
        self.n_of_features = len(training_set[0])
        self.n_of_output = len(self.labels[0])
        
        self.epochs = epochs
        self.layers = []
        self.task = task

        
        assert(batchSize > 0)

        if self.kind == 'SGD':
            self.training_algorithm = MiniBatchLearning(batchSize)
        elif self.kind == 'L-BFGS':
            self.training_algorithm = LBFGSTraining(batchSize)
            print("Sto usando LBFGS")

        self.restart = restart

        # Automatically add hidden / output layers
        for n in range(numHiddenLayers):
            self.addHiddenLayer(numOfUnitsPerLayer, weightsInitializer, activationFunction)

        self.addOutputLayer(numOfUnitPerOutput, task)

    def initializeWeights(self):
        for layer in self.layers:
            layer.initializeWeights()

    def addHiddenLayer(self, number_of_units, weightsInitializer='default', activation_function = 'sigmoid'):
        # If it's the first layer, the number of features is the training_set's column count
        if len(self.layers) == 0:
            number_of_features = self.training_set.shape[1]
        
        # If the network already has an output layer, do not add any new layer
        elif self.layers[-1].type == 'output':
            return
        
        # Generic case: the number fo features is give from the preceeding layer
        else:
            number_of_features = self.layers[-1].weights.shape[1]

        if self.kind == 'SGD':
            self.layers.append(
                Layer(
                    number_of_units,
                    number_of_features,
                    self.loss_function,
                    self.regularization_type,
                    self.regularization_lambda,
                    'hidden',
                    momentumAlpha=self.momentumAlpha,
                    momentumBeta=self.momentumBeta,
                    weightsInitializer = weightsInitializer,
                    activationFunction = activation_function
                    ) )
        elif self.kind == 'L-BFGS':
            self.layers.append(
                LBFGSLayer(
                    number_of_units,
                    number_of_features,
                    self.loss_function,
                    self.regularization_type,
                    self.regularization_lambda,
                    'hidden',
                    momentumAlpha=self.momentumAlpha,
                    momentumBeta=self.momentumBeta,
                    weightsInitializer = weightsInitializer,
                    activationFunction = activation_function
                    ) )

    def addOutputLayer(self, number_of_units = -1, task='classification'):
        if self.task == 'classification':
            activation_function = 'sigmoid'
        else:
            activation_function = 'linear'

        if number_of_units == -1:
            number_of_units = self.labels.shape[1]
        
        if ( self.layers[-1].type == 'hidden' ):
            if self.kind == 'SGD':
                self.layers.append(
                    Layer(
                        number_of_units,
                        (self.layers[-1].weights.shape[1]),
                        self.loss_function,
                        self.regularization_type,
                        self.regularization_lambda,
                        'output',
                        momentumAlpha=self.momentumAlpha,
                        momentumBeta=self.momentumBeta,
                        activationFunction = activation_function
                        ))
            elif self.kind == 'L-BFGS':
                self.layers.append(
                    LBFGSLayer(
                        number_of_units,
                        (self.layers[-1].weights.shape[1]),
                        self.loss_function,
                        self.regularization_type,
                        self.regularization_lambda,
                        'output',
                        momentumAlpha=self.momentumAlpha,
                        momentumBeta=self.momentumBeta,
                        activationFunction = activation_function
                        ))

    def predict(self, data):
        for layer in self.layers:
            data = layer.evaluate_input(data)
        return data

    def train(self, minimumVariation, plot=False, validation_set = np.array([]), validation_labels = np.array([]), test_set_for_plot=np.array([]), test_label_for_plot=np.array([])):
        if test_label_for_plot.size == 0 and test_set_for_plot.size == 0:
            error, epochs, error_on_validation, accuracy_mee, accuracy_mee_tr = self.training_algorithm.train(self.layers,
                                                        self.training_set,
                                                        self.labels,
                                                        self.learning_rate,
                                                        self.loss_function,
                                                        self.epochs,
                                                        self.test,
                                                        minimumVariation,
                                                        validation_set = validation_set,
                                                        validation_labels = validation_labels)

            if (self.restart > 1):
                best_epochs = epochs
                best_error = error
                best_layers = deepcopy(self.layers)
                best_error_valid = error_on_validation
                best_accuracy_mee = accuracy_mee
                best_accuracy_mee_tr = accuracy_mee_tr
                for _ in range(self.restart):
                    self.initializeWeights()

                    self.training_algorithm.reset()
                    error, epochs, error_on_validation, accuracy_mee, accuracy_mee_tr = self.training_algorithm.train(self.layers,
                                                            self.training_set,
                                                            self.labels,
                                                            self.learning_rate,
                                                            self.loss_function,
                                                            self.epochs,
                                                            self.test,
                                                            minimumVariation,
                                                            validation_set = validation_set,
                                                            validation_labels = validation_labels)
                    
                    if (error[-1] < best_error[-1]):
                        best_epochs = epochs
                        best_layers = deepcopy(self.layers)
                        best_error = error
                        best_error_valid = error_on_validation
                        best_accuracy_mee = accuracy_mee
                        best_accuracy_mee_tr = accuracy_mee_tr

                
                self.layers = deepcopy(best_layers)
                epochs = best_epochs
                error = best_error
                error_on_validation = best_error_valid
                accuracy_mee = best_accuracy_mee
                accuracy_mee_tr = best_accuracy_mee_tr

            if plot == True:
                plot_error_curve(epochs, error, error_on_validation)
                if self.task == 'regression':
                    label = "MEE"
                else:
                    label = "Accuracy"
                plot_accuracy_mee(accuracy_mee, accuracy_mee_tr, label)
            return error, error_on_validation

        else:
            error, epochs, error_on_test, accuracy_mee, accuracy_mee_tr = self.training_algorithm.train(
                self.layers,
                self.training_set,
                self.labels,
                self.learning_rate,
                self.loss_function,
                self.epochs,
                self.test,
                minimumVariation,
                validation_set=validation_set,
                validation_labels=validation_labels,
                test_set = test_set_for_plot,
                test_labels = test_label_for_plot
            )

            if (self.restart > 1):
                
                best_epochs = epochs
                best_error = error
                best_layers = deepcopy(self.layers)
                best_error_on_test = error_on_test
                best_accuracy_mee = accuracy_mee
                best_accuracy_mee_tr = accuracy_mee_tr
                for _ in range(self.restart):
                    self.initializeWeights()
                    self.training_algorithm.reset()
                    error, epochs, error_on_test, accuracy_mee, accuracy_mee_tr = self.training_algorithm.train(
                        self.layers,
                        self.training_set,
                        self.labels,
                        self.learning_rate,
                        self.loss_function,
                        self.epochs,
                        self.test,
                        minimumVariation,
                        validation_set=validation_set,
                        validation_labels=validation_labels,
                        test_set = test_set_for_plot,
                        test_labels = test_label_for_plot
                    )
                    
                    if (error[-1] < best_error[-1]):
                        best_epochs = epochs
                        best_layers = deepcopy(self.layers)
                        best_error = error
                        best_error_on_test = error_on_test
                        best_accuracy_mee = accuracy_mee
                        best_accuracy_mee_tr = accuracy_mee_tr

                
                self.layers = deepcopy(best_layers)
                epochs = best_epochs
                error = best_error
                error_on_test = best_error_on_test
                accuracy_mee = best_accuracy_mee
                accuracy_mee_tr = best_accuracy_mee_tr

            if plot == True:
                plot_error_curve(epochs, error, error_on_test)
                if self.task == 'regression':
                    label = "MEE"
                else:
                    label = "Accuracy"
                plot_accuracy_mee(accuracy_mee, accuracy_mee_tr, label)
            return error, error_on_test

    def test(self, test_set, labels):
        if self.task == 'classification':
            metric = Accuracy()
        else:
            metric = MEE()

        return metric(test_set, labels, self)



