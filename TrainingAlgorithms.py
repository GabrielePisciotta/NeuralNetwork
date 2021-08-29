import sys

import numpy as np
from sklearn.utils import shuffle
from Layer import *
from typing import List
import random
import math
import copy


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
            α = self.epoch / 200
            eta_t = eta_0 / 100
            learning_rate = (1 - α) * eta_0 + α * eta_t
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
                    gradient = layer.backward(accumulated_delta)

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


    # LINE SEARCH ALGORITHM FOR THE WOLFE CONDITIONS
    '''
    The parameter α_max is a user-supplied bound on the maximum step length allowed.
    '''
    def lineSearch(self, currNetworkk, c1=0.001, c2=0.9):
        currNetwork = [copy.deepcopy(c) for c in currNetworkk]

        alpha_0 = 0
        alpha_max = 1  # α_max > 0
        currAlpha = random.uniform(alpha_0, alpha_max)  # α_1 ∈ (0, α_max)

        initialDirDotGrad = self.computeDirectionDescent(currNetwork)

        phi0 = self.lineSearchEvaluate(0, currNetwork)
        prevAlpha = alpha_0

        phiPrevAlpha = np.finfo(np.float64).max
        for i in range(100):
            print("\tNel for... ", i)
            phiCurrAlpha = self.lineSearchEvaluate(currAlpha, currNetwork)
            if (phiCurrAlpha > phi0 + c1 * currAlpha * initialDirDotGrad) or (
                    i > 1 and phiCurrAlpha >= phiPrevAlpha):
                print("\t\tReturn zoom 1")
                return self.zoom(currNetwork, c1, c2, prevAlpha, currAlpha, phi0,
                            initialDirDotGrad)

            currDirDotGrad = self.computeDirectionDescent(currNetwork)

            if (abs(currDirDotGrad) <= - c2 * initialDirDotGrad):
                print("\t\tReturn currAlpha")
                return currAlpha
            if (currDirDotGrad >= 0):
                print("\t\tReturn zoom 2")
                return self.zoom(currNetwork, c1, c2, currAlpha, prevAlpha, phi0,
                            initialDirDotGrad)
            phiPrevAlpha = phiCurrAlpha
            prevAlpha = currAlpha
            currAlpha = random.uniform(prevAlpha, alpha_max)

        print("\t\tReturn finale random")
        return currAlpha

    def lineSearchEvaluate(self, stepSize, l):
        if stepSize == []:
            stepSize = 0

        # copy the layers in order to avoid changing of their internal values when computing the error
        # (by changing the weights)
        layers = [copy.deepcopy(ll) for ll in l]

        # Get the previously stored parameters
        training_set, labels, loss_function, TRLen = self.p

        ################
        # BACKWARD PHASE
        ################
        # After we've finished the feed forward phase, we calculate the delta of each layer
        # and update weights.
        accumulated_gradient = labels

        for layer in reversed(layers):

            # Compute gradient
            gradient = layer.backward(accumulated_gradient)

            # The following is needed in the following step of the backward propagation
            accumulated_gradient = gradient @ layer.weights.T

            # Get the previously computed direction
            direction = layer.direction

            # Update weights
            layer.weights, layer.bias = layer.weights_updater.update(layer.weights,
                                                                     layer.bias,
                                                                     layer.input,
                                                                     -direction,
                                                                     stepSize,
                                                                     )


        data = training_set
        for layer in layers:
            data = layer.evaluate_input(data)

        # Save the error on the training set for the graph
        return np.linalg.norm(loss_function.loss(labels, data)) * TRLen



    def train(self, layers, training_set, labels, learning_rate, loss_function, number_of_iterations, testFunction,
              minimumVariation, validation_set=np.array([]), validation_labels=np.array([]), test_set=np.array([]),
              test_labels=np.array([])):
        error_on_trainingset = []
        error_on_validationset = []
        accuracy_mee = []
        accuracy_mee_tr = []

        m = 10

        TRLen = training_set.shape[0]  # Size of the whole training set
        batchRanges = range(self.batchSize, TRLen + 1, self.batchSize)

        eta_0 = learning_rate
        self.epoch = 0
        self.grad = 1
        while True:
            print("Sto all'epoch ", self.epoch)

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
                for idx, layer in enumerate(reversed(layers)):

                    # Save old gradient@weights.T
                    q_old = layer.getGradientWeight().copy()

                    # Get the backward resulting gradient
                    gradient = layer.backward(accumulated_delta)

                    # The following is needed in the following step of the backward propagation
                    accumulated_delta = gradient @ layer.weights.T


                    # Compute the new input.T@gradient
                    layer.computeGradientWeight()
                    q = layer.getGradientWeight()

                    # Compute the direction -H_{k} ∇f_{k} (Algorithm 7.4 from the book)
                    # and store it for further usages (i.e.: find alpha!)
                    direction = -self.get_direction(layer)
                    layer.direction = direction

                    # Save the old weights
                    old_weights = layer.weights.copy()

                    # Find the proper step / learning rate (line search)
                    learning_rate = self.lineSearch(layers)#*np.sqrt(mb/TRLen)
                    print("\t\t\tLearning rate: ", learning_rate)

                    # Update weights
                    layer.weights, layer.bias = layer.weights_updater.update(layer.weights,
                                                                             layer.bias,
                                                                             layer.input,
                                                                             -direction,
                                                                             learning_rate)

                    # Create the list of the new curvature, taking into account that the first element is
                    # s_{k}= w_{k+1} - w_{k} while the second one is the y_{k} = ∇f_{k+1} - ∇f_{k}
                    s = layer.weights - old_weights
                    y = q-q_old

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

            # print("MEE/Accuracy on Train{}".format(accuracy_mee_tr[-1]))
            # print("MEE/Accuracy Valid{}".format(accuracy_mee[-1]))

            # Default stop condition
            StopCondition = True
            if (StopCondition):

                if self.epoch >= number_of_iterations:
                    print("Number of max iter reached")
                    break
                if err_tr <= np.finfo(np.float).eps:
                    print("Minimum gradient")
                    break

                if (self.epoch > 10):
                    diff = (error_on_trainingset[self.epoch] - error_on_trainingset[self.epoch - 1]) / \
                           error_on_trainingset[self.epoch]
                    if (abs(diff) < minimumVariation and abs(
                            error_on_trainingset[self.epoch]) < minimumVariation) or abs(
                            error_on_validationset[self.epoch]) < minimumVariation:
                        return error_on_trainingset, self.epoch, error_on_validationset, accuracy_mee, accuracy_mee_tr

            self.epoch += 1
        return error_on_trainingset, self.epoch, error_on_validationset, accuracy_mee, accuracy_mee_tr
    
    '''
    # Line-search conditions (page 33 from the CM book)
    # WOLFE CONDITIONS
    # 1. WOLFE CONDITION: Armijo condition, also called "sufficient decrease condition"
    def wolfe_armijo_condition(self, f, g, xk, alpha, pk):
        c1 = 1e-4
        return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)

    # 2. WOLFE CONDITION: Curvature condition
    def wolfe_curvature_condition(self, f, g, xk, alpha, pk):
        c1 = 1e-4
        return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)

    # STRONG WOLFE CONDITION
    def strong_wolfe(self, f, g, xk, alpha, pk, c2):
        return self.wolfe_armijo_condition(f, g, xk, alpha, pk) and abs(np.dot(g(xk + alpha * pk), pk)) <= c2 * abs(
            np.dot(g(xk), pk))
    '''
    #TODO capire dove chiamare il metodo
    # SECANT EQUATION
    '''
       s_{k}^T y_{k}>0      (6.7)
    '''
    def secantEquation(self, layer):
        if len(layer.past_curvatures) > 0:
            s, y = layer.past_curvatures[-1]
            if(np.dot(s.ravel().T, y.ravel()) <= 0):
                print("[ERROR] Secant equation not satisfied")
                return False
            else:
                return True
        else:
            return True

            # STEP-LENGTH SELECTION ALGORITHM - INTERPOLATION pag 56
    def quadraticApproximation(self, alphaLow, phiAlphaLo, searchDirectionDotGradientAlphaLow, alphaHi, phiAlphaHi):
        return -(searchDirectionDotGradientAlphaLow * alphaHi ** 2) / (
                2 * (phiAlphaHi - phiAlphaLo - searchDirectionDotGradientAlphaLow * alphaHi))

    def cubicApproximation(self, alphaLow, phiAlphaLow, searchDirectionDotGradientAlphaLow, alphaHi, phiAlphaHi,
                           searchDirectionDotGradientAlphaHi):
        d1 = searchDirectionDotGradientAlphaLow + searchDirectionDotGradientAlphaHi - 3 * (phiAlphaLow - phiAlphaHi) / (
                alphaLow - alphaHi)
        d2 = (1 if np.signbit(alphaHi - alphaLow) else -1) * math.sqrt(
            d1 ** 2 - searchDirectionDotGradientAlphaLow * searchDirectionDotGradientAlphaHi)
        return alphaHi - (alphaHi - alphaLow) * ((searchDirectionDotGradientAlphaHi + d2 - d1) / (searchDirectionDotGradientAlphaHi - searchDirectionDotGradientAlphaLow + 2 * d2))
    
    
    '''
    Compute dot product between the gradients store inside the layers \phi'
    '''
    def computeDirectionDescent(self, currNetwork):
        searchDirectionDotGradient = 0
        for currentLayer in currNetwork:
            grad = currentLayer.getGradientWeight().ravel()
            dir = currentLayer.GetDirection().ravel()
            searchDirectionDotGradient += np.dot(dir.T, grad)
        return searchDirectionDotGradient


    def zoom(self, currNetworkk, c1, c2, alphaLow, alphaHi, phi0, initialDirDotGrad):
        currNetwork = [copy.deepcopy(c) for c in currNetworkk]
        i = 0
        alphaJ = 1

        # limit number of iteration to obtain a step length in a finite time
        while (i < 100):
            # Compute \phi(\alpha_{j})
            phiCurrAlphaJ = self.lineSearchEvaluate(alphaJ, currNetwork)

            # Compute \phi(\alpha_{lo})
            phiCurrAlphaLow = self.lineSearchEvaluate(alphaLow, currNetwork)
            currDirDotGradAlphaLow = self.computeDirectionDescent(currNetwork)

            # Compute \alpha_{hi}
            phiCurrAlphaHi = self.lineSearchEvaluate(alphaHi, currNetwork)
            currDirDotGradAlphaHi = self.computeDirectionDescent(currNetwork)

            # quadraticInterpolation
            if phiCurrAlphaJ > (phi0 + c1 * alphaJ * initialDirDotGrad):
                alphaJ = self.quadraticApproximation(alphaLow,
                                                phiCurrAlphaLow,
                                                currDirDotGradAlphaLow,
                                                alphaHi,
                                                phiCurrAlphaHi)
                phiCurrAlphaJ = self.lineSearchEvaluate(alphaJ, currNetwork)

            # cubicInterpolation
            if phiCurrAlphaJ > (phi0 + c1 * alphaJ * initialDirDotGrad):
                alphaCubicInter = self.cubicApproximation(alphaLow, phiCurrAlphaLow,
                                                     currDirDotGradAlphaLow, alphaHi,
                                                     phiCurrAlphaHi,
                                                     currDirDotGradAlphaHi)

                if alphaCubicInter > 0 and alphaCubicInter <= 1:
                    alphaJ = alphaCubicInter
                    phiCurrAlphaJ = self.lineSearchEvaluate(alphaJ, currNetwork)

            # Bisection interpolation if quadratic goes wrong
            if alphaJ == 0:
                alphaJ = alphaLow + (alphaHi - alphaLow) / 2
                phiCurrAlphaJ = self.lineSearchEvaluate(alphaJ, currNetwork)

            if phiCurrAlphaJ > (phi0 + c1 * alphaJ * initialDirDotGrad) or phiCurrAlphaJ >= phiCurrAlphaLow:
                alphaHi = alphaJ
            else:
                # Compute \phi'(\alpha_{j})
                currDirDotGrad = self.computeDirectionDescent(currNetwork)

                if abs(currDirDotGrad) <= (-c2 * initialDirDotGrad):
                    return alphaJ

                if currDirDotGrad * (alphaHi - alphaLow) >= 0:
                    alphaHi = alphaLow
                alphaLow = alphaJ

            i = i + 1
        return alphaJ
