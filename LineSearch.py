import sys

import numpy as np
import random
import math
import copy

from sklearn.utils import shuffle
from typing import List

from Layer import *

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
    def __init__(self, currNetwork, params, c1=0.001, c2=0.9):
        self.currNetwork = currNetwork
        currNetwork_0 = self.getNetworkCopy(self.currNetwork)

        self.c1 = c1
        self.c2 = c2
        self.params = params
        self.alpha_0 = 0
        self.alpha_max = 1  # α_max > 0
        self.currAlpha = random.uniform(self.alpha_0, self.alpha_max)  # α_1 ∈ (0, α_max)

        self.initialDirDotGrad = self.computeDirectionDescent(currNetwork_0)
        self.phi0 = self.lineSearchEvaluate(0, currNetwork_0)
        self.phiPrevAlpha = np.finfo(np.float64).max

    def getNetworkCopy(self, network):
        return [copy.deepcopy(c) for c in network]

    # LINE SEARCH ALGORITHM FOR THE WOLFE CONDITIONS
    def lineSearch(self):
        prevAlpha = self.alpha_0

        for i in range(100):
            NN_for_J =  self.getNetworkCopy(self.currNetwork)
            phiCurrAlpha = self.lineSearchEvaluate(self.currAlpha, NN_for_J)
            if (phiCurrAlpha > self.phi0 + self.c1 * self.currAlpha * self.initialDirDotGrad) or (
                    i > 1 and phiCurrAlpha >= self.phiPrevAlpha):

                return self.zoom(NN_for_J, self.c1, self.c2, prevAlpha, self.currAlpha, self.phi0,
                            self.initialDirDotGrad)

            currDirDotGrad = self.computeDirectionDescent(NN_for_J)

            if (abs(currDirDotGrad) <= - self.c2 * self.initialDirDotGrad):
                return self.currAlpha
            if (currDirDotGrad >= 0):
                return self.zoom(NN_for_J, self.c1, self.c2, self.currAlpha, prevAlpha, self.phi0,
                            self.initialDirDotGrad)
            self.phiPrevAlpha = phiCurrAlpha
            prevAlpha = self.currAlpha
            self.currAlpha = random.uniform(prevAlpha, self.alpha_max)

        return self.currAlpha

    def lineSearchEvaluate(self, stepSize, layers):

        #layers = [copy.deepcopy(ll) for ll in l]

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
                                                                     -direction,
                                                                     stepSize,
                                                                     True)


        data = training_set
        for layer in layers:
            data = layer.evaluate_input(data, True)

        # Save the error on the training set for the graph
        return np.linalg.norm(loss_function.loss(labels, data))


    # STEP-LENGTH SELECTION ALGORITHM - INTERPOLATION pag 56
    def quadraticApproximation(self, phiAlphaLo, searchDirectionDotGradientAlphaLow, alphaHi, phiAlphaHi):
        return -(searchDirectionDotGradientAlphaLow * alphaHi ** 2) / (
                2 * (phiAlphaHi - phiAlphaLo - searchDirectionDotGradientAlphaLow * alphaHi))

    def cubicApproximation(self, alphaLow, phiAlphaLow, searchDirectionDotGradientAlphaLow, alphaHi, phiAlphaHi,
                           searchDirectionDotGradientAlphaHi):

        d1 = searchDirectionDotGradientAlphaLow + searchDirectionDotGradientAlphaHi -\
             3 * (phiAlphaLow - phiAlphaHi) / (alphaLow - alphaHi)
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


    def zoom(self, currNetwork, c1, c2, alphaLow, alphaHi, phi0, initialDirDotGrad):
        i = 0
        alphaJ = 1

        """
        From the "Numerical Optimization" book:
             Therefore, the line search must include a stopping test if it cannot attain a lower function
            value after a certain number (typically, ten) of trial step lengths. Some procedures also
            stop if the relative change in x is close to machine precision, or to some user-speciﬁed
            threshold
        
        """
        while i < 7:# or alphaJ-alphaLow <= np.finfo(np.float64).eps:
            # Compute \phi(\alpha_{j})
            currNetworkJ = self.getNetworkCopy(currNetwork)
            phiCurrAlphaJ = self.lineSearchEvaluate(alphaJ, currNetworkJ)

            # Compute \phi(\acontinuelpha_{lo})
            currNetworkLow = self.getNetworkCopy(currNetwork)
            phiCurrAlphaLow = self.lineSearchEvaluate(alphaLow, currNetworkLow)
            currDirDotGradAlphaLow = self.computeDirectionDescent(currNetworkLow)

            # Compute \alpha_{hi}
            currNetworkHigh = self.getNetworkCopy(currNetwork)
            phiCurrAlphaHi = self.lineSearchEvaluate(alphaHi, currNetworkHigh)
            currDirDotGradAlphaHi = self.computeDirectionDescent(currNetworkHigh)

            # quadraticInterpolation
            """if phiCurrAlphaJ > (phi0 + c1 * alphaJ * initialDirDotGrad):
                alphaJ = self.quadraticApproximation(phiCurrAlphaLow,
                                                currDirDotGradAlphaLow,
                                                alphaHi,
                                                phiCurrAlphaHi)
                currNetworksearch = self.getNetworkCopy(currNetwork)
                phiCurrAlphaJ = self.lineSearchEvaluate(alphaJ, currNetworksearch)"""
            # Bisection interpolation if quadratic goes wrong
            #if alphaJ == 0:
            #    alphaJ = alphaLow + (alphaHi - alphaLow) / 2

            # cubicInterpolation
            if phiCurrAlphaJ > (phi0 + c1 * alphaJ * initialDirDotGrad):
                alphaCubicInter = self.cubicApproximation(alphaLow,
                                                          phiCurrAlphaLow,
                                                          currDirDotGradAlphaLow,
                                                          alphaHi,
                                                          phiCurrAlphaHi,
                                                          currDirDotGradAlphaHi)

                if alphaCubicInter > 0 and alphaCubicInter <= 1:
                    alphaJ = alphaCubicInter
                    currNetworksearch = self.getNetworkCopy(currNetwork)
                    phiCurrAlphaJ = self.lineSearchEvaluate(alphaJ, currNetworksearch)
                else:
                    alphaJ = 0


            """
            Citing the book "Numerical Optimization":

                If any α_{i} is either too
                close to its predecessor α_{i−1} or else too much smaller than α_{i−1},
                we reset α_i to α_{i−1}/2. This safeguard procedure ensures that we make
                reasonable progress on each iteration and that the ﬁnal α is not too small.
            """
            eps = np.finfo(np.float64).eps
            if alphaJ == 0 or abs(alphaJ - alphaLow) <= 0.001:#eps or alphaJ == alphaLow+eps or alphaJ == alphaLow-eps:
                #print("[INFO] Safeguard encountered: alphaJ==0->", alphaJ == 0, "|abs(alphaJ - alphaLow) <= 0.001-> ", abs(alphaJ - alphaLow) <= 0.01)
                alphaJ = alphaLow / 2
                currNetworksearch = self.getNetworkCopy(currNetwork)
                phiCurrAlphaJ = self.lineSearchEvaluate(alphaJ, currNetworksearch)

            # Now that alpha_j is found...
            if phiCurrAlphaJ > (phi0 + c1 * alphaJ * initialDirDotGrad) or phiCurrAlphaJ >= phiCurrAlphaLow:
                alphaHi = alphaJ
            else:
                # Compute \phi'(\alpha_{j})
                currNetworksearch = self.getNetworkCopy(currNetwork)
                currDirDotGrad = self.computeDirectionDescent(currNetworksearch)

                if abs(currDirDotGrad) <= (-c2 * initialDirDotGrad):
                    return alphaJ

                if currDirDotGrad * (alphaHi - alphaLow) >= 0:
                    alphaHi = alphaLow
                alphaLow = alphaJ


            i = i + 1
        return alphaJ