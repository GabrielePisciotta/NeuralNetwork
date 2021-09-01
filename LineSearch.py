import sys

import numpy as np
import random
import math
import copy

from sklearn.utils import shuffle
from typing import List

from Layer import *
np.seterr(all='raise')

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

# LINE SEARCH CLASS
class LineSearch():
    def __init__(self, network, params, c1=0.001, c2=0.9):
        self.network = network
        network__0 = self.getNetworkCopy(self.network)

        self.c1 = c1
        self.c2 = c2
        self.params = params
        self.alpha_0 = 0
        self.alpha_max = 1  # α_max > 0
        self.currAlpha = random.uniform(self.alpha_0, self.alpha_max)  # α_1 ∈ (0, α_max)

        self.initialDirDotGrad = self.computeDirectionDescent(network__0)
        self.phi0 = self.evaluate(0, network__0)
        self.phiPrevAlpha = np.finfo(np.float64).max

    def getNetworkCopy(self, network):
        return [copy.deepcopy(c) for c in network]

    # LINE SEARCH ALGORITHM FOR THE WOLFE CONDITIONS
    def lineSearch(self):
        prevAlpha = self.alpha_0

        for i in range(10):
            network_J =  self.getNetworkCopy(self.network)
            phiCurrAlpha = self.evaluate(self.currAlpha, network_J)
            if (phiCurrAlpha > self.phi0 + self.c1 * self.currAlpha * self.initialDirDotGrad) or (
                    i > 1 and phiCurrAlpha >= self.phiPrevAlpha):

                return self.zoom(network_J, self.c1, self.c2, prevAlpha, self.currAlpha,self.initialDirDotGrad)

            currDirDotGrad = self.computeDirectionDescent(network_J)

            if (abs(currDirDotGrad) <= - self.c2 * self.initialDirDotGrad):
                return self.currAlpha
            if (currDirDotGrad >= 0):
                return self.zoom(network_J, self.c1, self.c2, self.currAlpha, prevAlpha,self.initialDirDotGrad)

            self.phiPrevAlpha = phiCurrAlpha
            prevAlpha = self.currAlpha
            self.currAlpha = random.uniform(prevAlpha, self.alpha_max)

        return self.currAlpha

    def evaluate(self, stepSize, ll):

        layers = [copy.deepcopy(ll) for ll in ll]

        # Get the previously stored parameters
        training_set, labels, loss_function, TRLen = self.params

        ################
        # BACKWARD PHASE
        ################
        # After we've finished the feed forward phase, we calculate the delta of each layer
        # and update weights.
        accumulated_gradient = labels
        for layer in reversed(layers):

            # Compute gradient
            gradient = layer.backward(accumulated_gradient, True)

            # The following is needed in the following step of the backward propagation
            accumulated_gradient = gradient @ layer.weights.T

            # Get the previously computed direction
            direction = layer.direction

            # Update weights
            layer.weights, layer.bias = layer.weights_updater.update(layer.weights,
                                                                     layer.bias,
                                                                     layer.input,
                                                                     direction,
                                                                     stepSize,
                                                                     True)


        data = training_set
        for layer in layers:
            data = layer.evaluate_input(data, True)

        # Save the error on the training set for the graph
        return np.linalg.norm(loss_function.loss(labels, data))/len(training_set)


    # STEP-LENGTH SELECTION ALGORITHM - INTERPOLATION pag 56
    def quadraticApproximation(self, phiAlphaLo, searchDirectionDotGradientAlphaLow, alphaHi, phiAlphaHi):
        return -(searchDirectionDotGradientAlphaLow * alphaHi ** 2) / (
                2 * (phiAlphaHi - phiAlphaLo - searchDirectionDotGradientAlphaLow * alphaHi))

    def cubicApproximation(self, alphaLow, phiAlphaLow, searchDirectionDotGradientAlphaLow, alphaHi, phiAlphaHi,
                           searchDirectionDotGradientAlphaHi):
        try:
            d1 = searchDirectionDotGradientAlphaLow + searchDirectionDotGradientAlphaHi -\
                 3 * (phiAlphaLow - phiAlphaHi) / (alphaLow - alphaHi)
            d2 = (1 if np.signbit(alphaHi - alphaLow) else -1) * math.sqrt(
                d1 ** 2 - searchDirectionDotGradientAlphaLow * searchDirectionDotGradientAlphaHi)
            return alphaHi - (alphaHi - alphaLow) * ((searchDirectionDotGradientAlphaHi + d2 - d1) / (searchDirectionDotGradientAlphaHi - searchDirectionDotGradientAlphaLow + 2 * d2))
        except:
            return 0
    
    '''
    Compute dot product between the gradients store inside the layers \phi'
    '''
    def computeDirectionDescent(self, network):
        searchDirectionDotGradient = 0
        for currentLayer in network:
            grad = currentLayer.getGradientWeight().ravel()
            dir = currentLayer.GetDirection().ravel()
            searchDirectionDotGradient += np.dot(dir.T, grad)
        return searchDirectionDotGradient


    def zoom(self, network, c1, c2, alphaLow, alphaHi, initialDirDotGrad):
        i = 0
        alphaJ = 1

        """
        From the "Numerical Optimization" book:
             Therefore, the line search must include a stopping test if it cannot attain a lower function
            value after a certain number (typically, ten) of trial step lengths. Some procedures also
            stop if the relative change in x is close to machine precision, or to some user-speciﬁed
            threshold
        
        """
        for i in range(5):
            # Compute \phi(\alpha_{j})
            network_J = self.getNetworkCopy(network)
            phiCurrAlphaJ = self.evaluate(alphaJ, network_J)

            # Compute \phi(\acontinuelpha_{lo})
            network_Low = self.getNetworkCopy(network)
            phiCurrAlphaLow = self.evaluate(alphaLow, network_Low)
            currDirDotGradAlphaLow = self.computeDirectionDescent(network_Low)

            # Compute \alpha_{hi}
            network_High = self.getNetworkCopy(network)
            phiCurrAlphaHi = self.evaluate(alphaHi, network_High)
            currDirDotGradAlphaHi = self.computeDirectionDescent(network_High)

            # quadraticInterpolation
            """if phiCurrAlphaJ > (phi0 + c1 * alphaJ * initialDirDotGrad):
                alphaJ = self.quadraticApproximation(phiCurrAlphaLow,
                                                currDirDotGradAlphaLow,
                                                alphaHi,
                                                phiCurrAlphaHi)
                network_search = self.getNetworkCopy(network)
                phiCurrAlphaJ = self.evaluate(alphaJ, network_search)"""
            # Bisection interpolation if quadratic goes wrong
            #if alphaJ == 0:
            #    alphaJ = alphaLow + (alphaHi - alphaLow) / 2

            # cubicInterpolation
            if phiCurrAlphaJ > (self.phi0 + c1 * alphaJ * initialDirDotGrad):
                alphaCubicInter = self.cubicApproximation(alphaLow,
                                                          phiCurrAlphaLow,
                                                          currDirDotGradAlphaLow,
                                                          alphaHi,
                                                          phiCurrAlphaHi,
                                                          currDirDotGradAlphaHi)

                if alphaCubicInter > 0 and alphaCubicInter <= 1:
                    alphaJ = alphaCubicInter
                    network_J = self.getNetworkCopy(network)
                    phiCurrAlphaJ = self.evaluate(alphaJ, network_J)
                else:
                    alphaJ = 0


            """
            Citing the book "Numerical Optimization":

                If any α_{i} is either too
                close to its predecessor α_{i−1} or else too much smaller than α_{i−1},
                we reset α_i to α_{i−1}/2. This safeguard procedure ensures that we make
                reasonable progress on each iteration and that the ﬁnal α is not too small.
            """
            #eps = np.finfo(np.float64).eps
            if alphaJ == 0:# or abs(alphaJ - alphaLow) <= 0.001:#eps or alphaJ == alphaLow+eps or alphaJ == alphaLow-eps:
                #print("[INFO] Safeguard encountered: alphaJ==0->", alphaJ == 0, "|abs(alphaJ - alphaLow) <= 0.001-> ", abs(alphaJ - alphaLow) <= 0.01)
                alphaJ = alphaLow / 2
                network_J = self.getNetworkCopy(network)
                phiCurrAlphaJ = self.evaluate(alphaJ, network_J)

            # Now that alpha_j is found, check conditions
            if phiCurrAlphaJ > (self.phi0 + c1 * alphaJ * initialDirDotGrad) or phiCurrAlphaJ >= phiCurrAlphaLow:
                alphaHi = alphaJ
            else:
                # Compute \phi'(\alpha_{j})
                currDirDotGrad = self.computeDirectionDescent(network_J)

                if abs(currDirDotGrad) <= (-c2 * initialDirDotGrad):
                    return alphaJ

                if currDirDotGrad * (alphaHi - alphaLow) >= 0:
                    alphaHi = alphaLow
                alphaLow = alphaJ

        return alphaJ