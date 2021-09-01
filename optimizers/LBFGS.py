import sys
import numpy as np
from sklearn.utils import shuffle
from models.NeuralNetwork.Layer import *
import copy
from linesearches.linesearch import LineSearch

RandomState = 4200000

class LBFGSTraining():
    def __init__(self, batchSize):
        self.batchSize = batchSize  # Mini-Batch size
        self.epoch = 0

    def reset(self):
        self.epoch = 0

    def get_H_0(self, layer):
        # Since we build the approximation of the hessian starting from the identity matrix,
        # following the equation H_{k}^{0} = γ_{k}*I, where γ_{k} = \frac{s^T_{k-1}y_{k-1}}{y^T_{k-1}y_{k-1}}
        # (eq. 7.20 from the Numerical Optimization book)

        I = np.ones((layer.getGradientWeight().shape[0], layer.getGradientWeight().shape[1]))

        s, y = layer.past_curvatures[-1]
        s = s.ravel()
        y = y.ravel()

        γ = s.T @ y / y.T @ y

        return γ * I

    def get_direction(self, layer):
        # This method returns the direction -r = - H_{k} ∇ f_{k}

        q = layer.getGradientWeight()
        original_shape = copy.deepcopy(q.shape)

        # Flattened vector
        q = q.ravel()

        # We'll skip the first γ creation (we have not sufficient information from the past curvatures)
        if len(layer.past_curvatures) > 1:

            # From latest curvature to the first
            for s, y in reversed(layer.past_curvatures):
                s = s.ravel()
                y = y.ravel()

                ρ = 1 / (y.T @ s)
                α = ρ * (s.T @ q)
                q = q - α * y

            H_0 = self.get_H_0(layer)
            r = H_0 * q.reshape(original_shape)

            # From first curvature to the last
            for s, y in layer.past_curvatures:
                s = s.ravel()
                y = y.ravel()

                ρ = 1 / (y.T @ s)
                α = ρ * (s.T @ q)

                β = ρ * (y.T @ r.ravel())
                r = r.ravel() + s * (np.array(α) - np.array(β))

            return r.reshape(original_shape)

        else:
            return q.reshape(original_shape)

    def train(self, layers, training_set, labels, learning_rate, loss_function, number_of_iterations, testFunction,
              minimumVariation, validation_set=np.array([]), validation_labels=np.array([]), test_set=np.array([]),
              test_labels=np.array([])):
        error_on_trainingset = []
        error_on_validationset = []
        accuracy_mee = []
        accuracy_mee_tr = []

        m = 7

        min_norma_grad_loss = 5e-5  # TODO: parametrizzare
        min_loss = 1e-2

        TRLen = training_set.shape[0]  # Size of the whole training set
        batchRanges = range(self.batchSize, TRLen + 1, self.batchSize)

        self.epoch = 0
        self.grad = 1
        while True:

            training_set, labels = shuffle(training_set, labels, random_state=RandomState)
            TRBatches = np.array_split(training_set, batchRanges)
            labelBatches = np.array_split(labels, batchRanges)

            for sample, label in zip(TRBatches, labelBatches):
                # store parameters for line search
                self.p = sample, label, loss_function, TRLen

                mb = sample.shape[0]  # Size of the minibatch
                # Modified ETA based on mb
                rate_multiplier = mb / TRLen
                rate_multiplier = np.sqrt(rate_multiplier)

                if rate_multiplier == 0:  # Empty batch iteration?
                    continue

                """
                FORWARD PHASE
                """
                data = np.array(sample)
                for layer in layers:
                    data = layer.evaluate_input(data)


                """
                BACKWARD PHASE:
                """
                # Compute directions
                accumulated_delta = label
                for layer in reversed(layers):
                    # Get the backward resulting gradient
                    gradient = layer.backward(accumulated_delta)

                    # The following is needed in the following step of the backward propagation
                    accumulated_delta = gradient @ layer.weights.T

                    # Compute the new input.T@gradient
                    layer.computeGradientWeight()

                    # Compute the direction -H_{k} ∇f_{k} (Algorithm 7.4 from the book)
                    # and store it for further usages (i.e.: find alpha!)
                    direction = self.get_direction(layer)
                    layer.direction = direction

                #  Find step size
                learning_rate = LineSearch(layers, self.p).lineSearch()
                if learning_rate <= 0:
                    print("[ERROR] learning rate is < 0")
                    sys.exit()
                if learning_rate > 1:
                    print("[ERROR] learning rate is > 0")
                    sys.exit()

                for layer in reversed(layers):

                    # Save old gradient@weights.T
                    q_old = layer.getGradientWeight().copy()

                    # Save the old weights
                    old_weights = layer.weights.copy()

                    # Update weights
                    layer.weights, layer.bias = layer.weights_updater.update(layer.weights,
                                                                             layer.bias,
                                                                             layer.input,
                                                                             layer.direction,
                                                                             learning_rate)

                    # Create the list of the new curvature, taking into account that the first element is
                    # s_{k}= w_{k+1} - w_{k} while the second one is the y_{k} = ∇f_{k+1} - ∇f_{k}
                    s = layer.weights - old_weights

                    # Compute the new input.T@gradient
                    #layer.computeGradientWeight()
                    q = layer.getGradientWeight()

                    y = q - q_old

                    # Secant equation
                    if not self.secantEquation(layer):
                        sys.exit()

                    # If the norm of both s and y is greater enough, store it
                    if np.linalg.norm(s) > 1 and np.linalg.norm(y) > np.finfo(np.float64).eps:
                        layer.past_curvatures.append([s, y])

                    # Remove the oldest element in order to keep the list with the desired size (m)
                    if len(layer.past_curvatures) > m:
                        layer.past_curvatures.pop(0)

                    # Update k
                    layer.k += 1

            data = training_set
            for layer in layers:
                data = layer.evaluate_input(data)

            # Save the error on the training set for the graph
            err_tr = np.sum(loss_function.loss(labels, data)) / len(training_set)
            error_on_trainingset.append(err_tr)

            data = validation_set
            for layer in layers:
                data = layer.evaluate_input(data)

            # Save the error on the validation set for the  graph
            error_on_validationset.append(np.sum(loss_function.loss(validation_labels, data)) / len(validation_set))

            if test_labels.size == 0 and test_labels.size == 0:
                # Save the accuracy/mee on validation set for the graph
                accuracy_mee.append(testFunction(validation_set, validation_labels))
            else:
                # Save the accuracy/mee on test set for the graph
                accuracy_mee.append(testFunction(test_set, test_labels))

            # Save the accuracy/mee on test set for the graph
            accuracy_mee_tr.append(testFunction(training_set, labels))
            print("Epoch: ", self.epoch,
                  "\n\t TR error: ", error_on_trainingset[-1], " | VS error: ", error_on_validationset[-1],
                  "\n\t TR accuracy: ", accuracy_mee_tr[-1], " | VS error: ", accuracy_mee[-1])

            # print("MEE/Accuracy on Train{}".format(accuracy_mee_tr[-1]))
            # print("MEE/Accuracy Valid{}".format(accuracy_mee[-1]))

            # Default stop condition
            StopCondition = True
            if (StopCondition):

                if self.epoch >= number_of_iterations:
                    print("Number of max iter reached")
                    break
                if err_tr <= min_loss + np.finfo(np.float).eps:
                    print("Minimum loss")
                    break

                """if abs(np.linalg.norm(layers[-1].getGradientWeight())) <= min_norma_grad_loss + np.finfo(np.float).eps:
                    print("Minimum gradient getGradientWeight: ", abs(np.linalg.norm(layers[-1].getGradientWeight())))
                    break"""

                if (self.epoch > 10):
                    diff = (error_on_trainingset[self.epoch] - error_on_trainingset[self.epoch - 1]) / \
                           error_on_trainingset[self.epoch]
                    if (abs(diff) < minimumVariation and abs(
                            error_on_trainingset[self.epoch]) < minimumVariation) or abs(
                        error_on_validationset[self.epoch]) < minimumVariation:
                        return error_on_trainingset, self.epoch, error_on_validationset, accuracy_mee, accuracy_mee_tr

            self.epoch += 1
        return error_on_trainingset, self.epoch, error_on_validationset, accuracy_mee, accuracy_mee_tr

    # SECANT EQUATION
    '''
        s_{k}^T y_{k}>0      (6.7)
    '''

    def secantEquation(self, layer):
        if len(layer.past_curvatures) > 0:
            s, y = layer.past_curvatures[-1]
            if (np.dot(s.ravel().T, y.ravel()) <= 0):
                print("[ERROR] Secant equation not satisfied")
                return False
            else:
                return True
        else:
            return True