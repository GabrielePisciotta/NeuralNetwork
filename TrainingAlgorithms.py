import numpy as np
from sklearn.utils import shuffle
from Layer import *
from typing import List

RandomState = 4200000

class TrainingAlgorithm:
    def __init__(self): pass
    def train(self, layers, training_set, labels, learning_rate, loss_function, number_of_iterations, validation_set, validation_labels):
        pass


class MiniBatchLearning(TrainingAlgorithm):
    def __init__(self, batchSize):
        self.batchSize = batchSize # Mini-Batch size
        self.epoch = 0
        
    def reset(self):
        self.epoch = 0
    
    def train(self, layers, training_set, labels, learning_rate, loss_function, number_of_iterations, testFunction, minimumVariation, validation_set = np.array([]), validation_labels = np.array([]), test_set=np.array([]), test_labels=np.array([])):
        error_on_trainingset = []
        error_on_validationset = []
        accuracy_mee = []
        accuracy_mee_tr = []

        TRLen = training_set.shape[0] # Size of the whole training set
        batchRanges = range(self.batchSize, TRLen + 1, self.batchSize)

        eta_0 = learning_rate
        self.epoch = 0
        while self.epoch < number_of_iterations:
            # learning rate decay as stated in the slide
            alfa = self.epoch / 200
            eta_t = eta_0 / 100
            learning_rate = (1 - alfa) * eta_0 + alfa * eta_t
            if (self.epoch > 200):
                learning_rate = eta_t

            # time based decay (the 0.005 is a decay factor: to be optimized!)
            #learning_rate = eta_0 / (1. + 0.01 * self.epoch)

            training_set, labels = shuffle(training_set, labels, random_state = RandomState)
            
            TRBatches = np.array_split(training_set, batchRanges)
            labelBatches = np.array_split(labels, batchRanges)

            for sample, label in zip(TRBatches, labelBatches):
                mb = sample.shape[0] # Size of the minibatch
                # Modified ETA based on mb
                rate_multiplier = mb / TRLen
                rate_multiplier = np.sqrt(rate_multiplier)
                
                if rate_multiplier == 0: # Empty batch iteration?
                    continue
                
                ###############
                # FORWARD PHASE
                ###############
                # We can propagate the output of the first layer.
                # For each neuron of the layer, compute the output based on the
                # output of the precedent layer
                data = np.array(sample)
                for layer in layers:
                    data = layer.evaluate_input(data)

                ################
                # BACKWARD PHASE
                ################
                # After we've finished the feed forward phase, we calculate the delta of each layer
                # and update weights.
                accumulated_delta = label
                for layer in reversed(layers):
                    gradient = layer.backward(accumulated_delta, rate_multiplier * learning_rate)

                    # The following is needed in the following step of the backward propagation
                    accumulated_delta = gradient @ layer.weights.T

                    # Update weights
                    layer.weights, layer.bias = layer.weights_updater.update(layer.weights,
                                                                          layer.bias,
                                                                          layer.input,
                                                                          layer.delta,
                                                                          learning_rate)
            data = training_set
            for layer in layers:
                data = layer.evaluate_input(data)

            # Save the error on the training set for the graph
            error_on_trainingset.append(np.sum(loss_function.loss(labels, data)) / len(training_set))

            data = validation_set
            for layer in layers:
                data = layer.evaluate_input(data)

            # Save the error on the validation set for the  graph
            error_on_validationset.append(np.sum(loss_function.loss(validation_labels, data)) / len(validation_set))

            if test_labels.size == 0 and test_labels.size == 0:
                # Save the accuracy/mee on validation set for the graph
                accuracy_mee.append( testFunction(validation_set, validation_labels) )
            else:
                # Save the accuracy/mee on test set for the graph
                accuracy_mee.append(testFunction(test_set, test_labels))

            # Save the accuracy/mee on test set for the graph
            accuracy_mee_tr.append(testFunction(training_set, labels))

            #print("MEE/Accuracy on Train{}".format(accuracy_mee_tr[-1]))
            #print("MEE/Accuracy Valid{}".format(accuracy_mee[-1]))

            # Default stop condition
            StopCondition = True
            if (StopCondition):
                if (self.epoch > 10):
                    diff = (  error_on_trainingset[self.epoch] - error_on_trainingset[self.epoch-1]) / error_on_trainingset[self.epoch]
                    if ( abs(diff) < minimumVariation and abs(error_on_trainingset[self.epoch]) < minimumVariation) or abs(error_on_validationset[self.epoch]) < minimumVariation:
                        return error_on_trainingset, self.epoch, error_on_validationset, accuracy_mee, accuracy_mee_tr

            self.epoch += 1
        return error_on_trainingset, self.epoch, error_on_validationset, accuracy_mee, accuracy_mee_tr


class LBFGSTraining(TrainingAlgorithm):
    def __init__(self, batchSize):
        self.batchSize = batchSize  # Mini-Batch size
        self.epoch = 0

    def reset(self):
        self.epoch = 0

    def get_direction(self, layer):

        # TODO: Check secant equation conditions (s_{k}^T y_{k} > 0) if layer.s[k].T @ layer.y[k] > 0:
        q_old = layer.accumulated_gradient.copy()

        q = layer.input.T @ layer.delta
        layer.accumulated_gradient = q

        y = q-q_old

        # Since we build the approximation of the hessian starting from the identity matrix,
        # following the equation H_{k}^{0} = \gamma_{k}*I, where \gamma_{k} = \frac{s^T_{k-1}y_{k-1}}{y^T_{k-1}y_{k-1}}
        # (eq. 7.20 from the Numerical Optimization book)
        H = np.eye(q.shape[0], q.shape[1])

        # We'll skip the first gamma creation (we have not sufficient information from the past curvatures)
        if len(layer.past_curvatures) > 0:
            s = layer.past_curvatures[-1][0]
            y = layer.past_curvatures[-1][1]
            gamma = (s - y)/(y*y)
            """gamma = np.asscalar(
                (layer.past_curvatures[-1][0] * layer.past_curvatures[-1][1]) # s^T_{k-1}y_{k-1}
                /
                (layer.past_curvatures[-1][1] * layer.past_curvatures[-1][1]) # y^T_{k-1}y_{k-1}
            )"""
            H *= gamma  # \gamma_{k}*I

        # Variables used to store ρ and α
        list_ρ = []
        list_α = []

        # Iterate through the latest past curvatures available (tail to head)
        for past_curvature in layer.past_curvatures:

            ρ = 1 / past_curvature[1] * past_curvature[0]
            α = ρ * past_curvature[0] * q
            q -= α * past_curvature[1]

            # Save rho and alpha for further usages in the next loop
            list_ρ.insert(0, ρ)
            list_α.insert(0, α)

        r = H * q

        # Iterate through the latest past curvatures available (head to tail)
        for idx, past_curvature in enumerate(reversed(layer.past_curvatures)):
            α = list_α[idx]
            ρ = list_ρ[idx][0]

            β = ρ * past_curvature[1] * r

            parentesi = (α - β)
            r += past_curvature[0] * parentesi
        # End of 7.4

        return -r, y

    def train(self,  layers: List[LBFGSLayer], training_set, labels, learning_rate, loss_function, number_of_iterations, testFunction,
              minimumVariation, validation_set=np.array([]), validation_labels=np.array([]), test_set=np.array([]),
              test_labels=np.array([])):
        error_on_trainingset = []
        error_on_validationset = []
        accuracy_mee = []
        accuracy_mee_tr = []

        m = 7# TODO: parametrize this
        TRLen = training_set.shape[0]  # Size of the whole training set
        batchRanges = range(self.batchSize, TRLen + 1, self.batchSize)

        eta_0 = learning_rate
        self.epoch = 0
        while self.epoch < number_of_iterations:
            # learning rate decay as stated in the slide
            alfa = self.epoch / 200
            eta_t = eta_0 / 100
            learning_rate = (1 - alfa) * eta_0 + alfa * eta_t
            if (self.epoch > 200):
                learning_rate = eta_t

            # time based decay (the 0.005 is a decay factor: to be optimized!)
            # learning_rate = eta_0 / (1. + 0.01 * self.epoch)

            training_set, labels = shuffle(training_set, labels, random_state=RandomState)

            TRBatches = np.array_split(training_set, batchRanges)
            labelBatches = np.array_split(labels, batchRanges)

            for sample, label in zip(TRBatches, labelBatches):
                mb = sample.shape[0]  # Size of the minibatch
                # Modified ETA based on mb
                rate_multiplier = mb / TRLen
                rate_multiplier = np.sqrt(rate_multiplier)

                if rate_multiplier == 0:  # Empty batch iteration?
                    continue

                ###############
                # FORWARD PHASE
                ###############
                # We can propagate the output of the first layer.
                # For each neuron of the layer, compute the output based on the
                # output of the precedent layer
                data = np.array(sample)
                for layer in layers:
                    data = layer.evaluate_input(data)

                ################
                # BACKWARD PHASE
                ################
                # After we've finished the feed forward phase, we calculate the delta of each layer
                # and update weights.
                accumulated_gradient = label

                n_of_layers = 0
                for layer in reversed(layers):
                    k = layer.k

                    # Save the old weights
                    old_weights = layer.weights.copy()

                    # Compute gradient
                    gradient, old_gradient = layer.backward(accumulated_gradient, rate_multiplier * learning_rate)

                    # The following is needed in the following step of the backward propagation
                    accumulated_gradient = gradient @ layer.weights.T

                    # Compute the direction (Algorithm 7.4 from the book, to compute -H ∇f
                    direction, y = self.get_direction(layer)

                    # Update weights
                    layer.weights, layer.bias, s = layer.weights_updater.update(layer.weights,
                                                                             layer.bias,
                                                                             layer.input,
                                                                             direction,
                                                                             100*learning_rate)

                    # Create the list of the new curvature, taking into account that the first element is
                    # s_{k}= w_{k+1} - w_{k} while the second one is the y

                    if layer.k in layer.past_curvatures:
                        layer.past_curvatures[layer.k] = [s, y]
                    else:
                        layer.past_curvatures.insert(layer.k,  [s, y])

                    # Remove the oldest element in order to keep the list with the desired size (m)
                    if len(layer.past_curvatures) > m:
                        #print("[INFO] Removing old curvatures, at iter ", k)
                        layer.past_curvatures.pop(0)

                    if len(layer.past_curvatures) > m:
                        layer.s.pop(0)

                    # Update k
                    layer.k += 1

                    n_of_layers +=1

            data = training_set
            for layer in layers:
                data = layer.evaluate_input(data)

            # Save the error on the training set for the graph
            error_on_trainingset.append(np.sum(loss_function.loss(labels, data)) / len(training_set))

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

            # print("MEE/Accuracy on Train{}".format(accuracy_mee_tr[-1]))
            # print("MEE/Accuracy Valid{}".format(accuracy_mee[-1]))

            # Default stop condition
            StopCondition = True
            if (StopCondition):
                if (self.epoch > 10):
                    diff = (error_on_trainingset[self.epoch] - error_on_trainingset[self.epoch - 1]) / \
                           error_on_trainingset[self.epoch]
                    if (abs(diff) < minimumVariation and abs(
                            error_on_trainingset[self.epoch]) < minimumVariation) or abs(
                            error_on_validationset[self.epoch]) < minimumVariation:
                        return error_on_trainingset, self.epoch, error_on_validationset, accuracy_mee, accuracy_mee_tr

            self.epoch += 1
        return error_on_trainingset, self.epoch, error_on_validationset, accuracy_mee, accuracy_mee_tr
