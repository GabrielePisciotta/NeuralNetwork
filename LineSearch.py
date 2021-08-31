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
        self.currNetwork = [copy.deepcopy(c) for c in currNetwork]
        self.c1 = c1
        self.c2 = c2
        self.params = params
        self.alpha_0 = 0
        self.alpha_max = 1  # α_max > 0
        self.currAlpha = random.uniform(self.alpha_0, self.alpha_max)  # α_1 ∈ (0, α_max)
        self.initialDirDotGrad = self.computeDirectionDescent(currNetwork)
        self.phi0 = self.lineSearchEvaluate(0, currNetwork) 
        self.phiPrevAlpha = np.finfo(np.float64).max
        

    # LINE SEARCH ALGORITHM FOR THE WOLFE CONDITIONS
    def lineSearch(self):   
        prevAlpha = self.alpha_0 
        for i in range(100):
            print("\tNel for... ", i)
            phiCurrAlpha = self.lineSearchEvaluate(self.currAlpha, self.currNetwork)
            if (phiCurrAlpha > self.phi0 + self.c1 * self.currAlpha * self.initialDirDotGrad) or (
                    i > 1 and phiCurrAlpha >= self.phiPrevAlpha):
                #print("\t\tReturn zoom 1")
                return self.zoom(self.currNetwork, self.c1, self.c2, prevAlpha, self.currAlpha, self.phi0,
                            self.initialDirDotGrad)

            currDirDotGrad = self.computeDirectionDescent(self.currNetwork)

            if (abs(currDirDotGrad) <= - self.c2 * self.initialDirDotGrad):
                #print("\t\tReturn currAlpha")
                return self.currAlpha
            if (currDirDotGrad >= 0):
                #print("\t\tReturn zoom 2")
                return self.zoom(self.currNetwork, self.c1, self.c2, self.currAlpha, prevAlpha, self.phi0,
                            self.initialDirDotGrad)
            self.phiPrevAlpha = phiCurrAlpha
            prevAlpha = self.currAlpha
            self.currAlpha = random.uniform(prevAlpha, self.alpha_max)

        #print("\t\tReturn finale random")
        return self.currAlpha

    def lineSearchEvaluate(self, stepSize, l):
        if stepSize == []:
            stepSize = 0

        # copy the layers in order to avoid changing of their internal values when computing the error
        # (by changing the weights)
        layers = [copy.deepcopy(ll) for ll in l]

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


    def zoom(self, currNetwork, c1, c2, alphaLow, alphaHi, phi0, initialDirDotGrad):
        currNetwork = [copy.deepcopy(c) for c in currNetwork]
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