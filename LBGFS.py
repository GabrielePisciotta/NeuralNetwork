import math
import random

import TrainingAlgorithms
from TrainingAlgorithms import *
import numpy as np

# Line-search conditions

# PAG 33 del libro
'''
  g: ∇fkT the initial gradient direction for the given initial value -> g(xk)
  xk: initial value
  alpha: step length 
  pk: descent direction
  c1: value between (0,1)
  c2: value between (c1,1)
'''

# WOLFE CONDITIONS
'''
f(xk + αk pk ) ≤ f(xk) + c1 αk ∇fkT pk       (3.6a)
∇f(xk + αk pk)T pk ≥ c2 ∇fkT pk              (3.6b)
'''

# WOLFE CONDITION: Armijo condition, also called "sufficient decrease condition"
'''
f (xk + αpk ) ≤ f(xk) + c1 α ∇fkT pk  (3.4)
-Note that: alpha is accettable only if φ(α) ≤ l(α). 
  l(α) is the the right-hand-side of (3.4) (linear function)

-Note that: c1 is chosen to be quite small, say c1 = 10^−4
'''


def wolfe_armijo_condition(f, g, xk, alpha, pk):
    c1 = 1e-4
    return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)


# WOLFE CONDITION: Curvature condition
'''
-Note that: The Armijo condition is not enough by itself to ensure that the algorithm makes reasonable 
  progress because, as we see from Figure 3.3, it is satisfied for all sufficiently small values of α. 
  To rule out unacceptably short steps we introduce a second requirement, called the curvature condition, 
  which requires αk to satisfy:
  ∇f(xk + αk pk)T pk ≥ c2 ∇fkT pk   (3.5)

-Note that the left-handside is simply the derivative φ'(αk), so the curvature condition ensures 
  that the slope of φ at αk is greater than c2 times the initial slope φ'(0). 
'''


def wolfe_curvature_condition(f, g, xk, alpha, pk):
    c1 = 1e-4
    return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)


# STRONG WOLFE CONDITION
'''
-A step length may satisfy the Wolfe conditions without being particularly close to a minimizer of φ. 
  We can, however, modify the curvature condition to force αk to lie in at least a broad neighborhood 
  of a local minimizer or stationary point of φ. The strong Wolfe conditions require αk to satisfy:
  f(xk + αk pk) ≤ f (xk) + c1 αk ∇fkT pk    (3.7a)
  |∇f(xk + αk pk)T pk| ≤ c2 |∇fkT pk|       (3.7b)
-The only difference with the Wolfe conditions is that we no longer allow the derivative φ'(αk) to be too positive. 
  Hence, we exclude points that are far from stationary points of φ.
'''


def strong_wolfe(f, g, xk, alpha, pk, c2):
    return wolfe_armijo_condition(f, g, xk, alpha, pk) and abs(np.dot(g(xk + alpha * pk), pk)) <= c2 * abs(
        np.dot(g(xk), pk))


# BACKTRACKING - line search (algo 3.1)
'''
-If the line search algorithm chooses its candidate step lengths appropriately, by using the "backtracking" approach, 
  we can dispense with the "curvature condition" and use just the sufficient decrease condition to terminate 
  the line search procedure. In its most basic form, backtracking proceeds as follows.
-In this procedure, the initial step length α¯ is chosen to be 1 in Newton and quasiNewton methods, but can have different values in other algorithms such as steepest descent
  or conjugate gradient.
'''


def backtracking(f, g, xk, pk):
    alpha_s = 1
    rho = random.randrange(0, 1, 0.01)
    c = random.randrange(0, 1, 0.0001)
    alpha = alpha_s

    while (f(xk + alpha * pk) <= f(xk) + c * alpha * np.dot(g(xk), pk)):
        alpha = alpha * rho

    return alpha


# STEP-LENGTH SELECTION ALGORITHM - INTERPOLATION pag 56
'''
-All line search procedures require an initial estimate α0 and generate a sequence {αi} that either terminates with a 
  step length satisfying the conditions specified by the user (for example, the Wolfe conditions) or determines that 
  such a step length does not exist. Typical procedures consist of two phases: a BRACKETING PHASE that finds an interval 
  [a¯, b¯] containing acceptable step lengths, and a SELECTION PHASE that zooms in to locate the final step length.
-The selection phase usually reduces the bracketing interval during its search for the desired step length and 
  interpolates some of the function and derivative information gathered on earlier steps to guess the location of the 
  minimizer. We first discuss how to perform this interpolation.
-In the following discussion we let αk and αk−1 denote the step lengths used at iterations k and k − 1 of the 
  optimization algorithm, respectively. On the other hand, we denote the trial step lengths generated during the line 
  search by αi and αi−1 and also αj . We use α0 to denote the initial guess.
-This procedure can be viewed as an enhancement of Algorithm 3.1. The aim is to find a value of α that satisfies the 
  sufficient decrease condition (3.6a), without being “too small.” Accordingly, the procedures here generate a decreasing 
  sequence of values αi such that each value αi is not too much smaller than its predecessor αi−1.

  φ(α) = f(xk + αpk)                      (3.54)
  f (xk + αpk) ≤ f(xk) + c1 α ∇fkT pk     (3.4)

  We can write the 3.4 with the notation of the 3.54:
    φ(αk) ≤ φ(0) + c1 αk φ'(0)
  
  Suppose that the initial guess α0 is given. If we have:
    φ(α0) ≤ φ(0) + c1 α0 φ'(0)
  this step length satisfies the condition, and we terminate the search. Otherwise, we know that the interval [0, α0] 
  contains acceptable step lengths. 
-We form a quadratic approximation φq(α) to φ by interpolating the three pieces of information available φ(0), φ'(0), 
  and φ(α0) to obtain:
    φq(α) = ( (φ(α0) − φ(0) − α0φ'(0) ) / α0^2) α^2 + φ'(0)α + φ(0)       (3.57)
  The new trial value α1 is defined as the minimizer of this quadratic, that is, we obtain:
    α1 = − ( (φ'(0)α0^2) / (2[φ(α0) − φ(0) − φ'(0)α0]) )                  (3.58)
  If the sufficient decrease condition (3.56) is satisfied at α1, we terminate the search. Otherwise, we construct a 
  cubic function that interpolates the four pieces of information φ(0), φ'(0), φ(α0), and φ(α1), obtaining:
    φc(α) = aα3 + bα2 + αφ'(0) + φ(0)
  where a and b are calculated (see book at page 58).
  By differentiating φc(x), we see that the minimizer α2 of φc lies in the interval [0, α1] and is given by:
    α2 = (-b + sqrt(b^2 - 3a φ'(0)) ) / 3a
  If necessary, this process is repeated, using a cubic interpolant of φ(0), φ'(0) and the two most recent values of φ, 
  until an α that satisfies (3.56) is located. If any αi is either too close to its predecessor αi−1 or else too much 
  smaller than αi−1, we reset αi  αi−1/2. This safeguard procedure ensures that we make reasonable progress on each 
  iteration and that the final α is not too small.

  -Cubic interpolation provides a good model for functions with significant changes of curvature. Suppose we have an 
  interval [a¯, b¯] known to contain desirable step lengths, and two previous step length estimates αi−1 and αi in this 
  interval. We use a cubic function to interpolate φ(αi−1), φ'(αi−1), φ(αi), and φ'(αi). The minimizer of this cubic in
  [a¯, b¯] is either at one of the endpoints or else in the interior, in which case it is given by the 3.59 (page 59).
    CHECK BOOK pag 59
  
  -For Newton and quasi-Newton methods, the step α0=1 should always be used as the initial trial step length.
'''


def quadraticApproximation(alphaLow, phiAlphaLo, searchDirectionDotGradientAlphaLow, alphaHi, phiAlphaHi):
    return -(searchDirectionDotGradientAlphaLow * alphaHi ** 2) / (
                2 * (phiAlphaHi - phiAlphaLo - searchDirectionDotGradientAlphaLow * alphaHi))


def cubicApproximation(alphaLow, phiAlphaLow, searchDirectionDotGradientAlphaLow, alphaHi, phiAlphaHi,
                       searchDirectionDotGradientAlphaHi):
    d1 = searchDirectionDotGradientAlphaLow + searchDirectionDotGradientAlphaHi - 3 * (phiAlphaLow - phiAlphaHi) / (
                alphaLow - alphaHi)
    d2 = (1 if np.signbit(alphaHi - alphaLow) else -1) * math.sqrt(
        d1 ** 2 - searchDirectionDotGradientAlphaLow * searchDirectionDotGradientAlphaHi)
    return alphaHi - (alphaHi - alphaLow) * ((
                                                         searchDirectionDotGradientAlphaHi + d2 - d1) / searchDirectionDotGradientAlphaHi - searchDirectionDotGradientAlphaLow + 2 * d2)


'''
Compute dot product between the gradients store inside the layers \phi'
'''


def computeDirectionDescent(currNetwork):
    searchDirectionDotGradient = 0
    for currentLayer in currNetwork:
        primo = currentLayer.getGradientWeight()
        secondo =  currentLayer.GetDirection()
        scal = np.dot(primo.T, secondo)
        searchDirectionDotGradient += scal
    return searchDirectionDotGradient


# LINE SEARCH ALGORITHM FOR THE WOLFE CONDITIONS
'''
The parameter α_max is a user-supplied bound on the maximum step length allowed.
'''


def lineSearch(currNetwork, params, c1=0.001, c2=0.9):
    alpha_0 = 0
    alpha_max = 0.99  # α_max > 0
    currentAlpha = random.uniform(alpha_0, alpha_max)  # α_1 ∈ (0, α_max)

    initialSearchDirectionDotGradient = computeDirectionDescent(currNetwork)

    # Check descent direction
    if (initialSearchDirectionDotGradient.any() > 0.0):  #TODO Rimuovere porcata
        return False

    phi0 = self.lineSearchEvaluation([], params)
    
    previousAlpha = alpha_0

    phiPreviousAlpha = 999999999 #TODO cambiare porcata
    for i in range(100):
        phiCurrentAlpha = self.lineSearchEvaluation(currentAlpha, params)
        if ((phiCurrentAlpha > phi0 + c1 * currentAlpha * initialSearchDirectionDotGradient) or (
                i > 1 and phiCurrentAlpha >= phiPreviousAlpha)):
            return zoom(currNetwork, c1, c2, previousAlpha, currentAlpha, phi0,
                        initialSearchDirectionDotGradient)

        currentSearchDirectionDotGradient = computeDirectionDescent(currNetwork)

        if (abs(currentSearchDirectionDotGradient) <= c2 * initialSearchDirectionDotGradient):
            return currentAlpha
        if (currentSearchDirectionDotGradient >= 0):
            return zoom(currNetwork, c1, c2, currentAlpha, previousAlpha, phi0,
                        initialSearchDirectionDotGradient)
        phiPreviousAlpha = phiCurrentAlpha
        previousAlpha = currentAlpha
        currentAlpha = random.uniform(previousAlpha, alpha_max)
    return currentAlpha


def zoom(currNetwork, c1, c2, alphaLow, alphaHi, phi0, initialSearchDirectionDotGradient, params):
    tr = TrainingAlgorithms.LBFGSTraining
    i = 0
    alphaJ = 1

    # limit number of iteration to obtain a step length in a finite time
    while (i < 100):
        # Compute \phi(\alpha_{j})
        phiCurrentAlphaJ =self.lineSearchEvaluation(alphaJ, params)

        # Compute \phi(\alpha_{lo})
        phiCurrentAlphaLow =self.lineSearchEvaluation(alphaLow, params)
        currentSearchDirectionDotGradientAlphaLow = computeDirectionDescent(currNetwork)

        # Compute \alpha_{hi}
        phiCurrentAlphaHi =self.lineSearchEvaluation(alphaHi, params)
        currentSearchDirectionDotGradientAlphaHi = computeDirectionDescent(currNetwork)

        # quadraticInterpolation
        if (phiCurrentAlphaJ > phi0 + c1 * alphaJ * initialSearchDirectionDotGradient):
            alphaJ = quadraticApproximation(alphaLow,
                                            phiCurrentAlphaLow,
                                            currentSearchDirectionDotGradientAlphaLow,
                                            alphaHi,
                                            phiCurrentAlphaHi)
            phiCurrentAlphaJ =self.lineSearchEvaluation(alphaJ, params)

        # cubicInterpolation
        if (phiCurrentAlphaJ > phi0 + c1 * alphaJ * initialSearchDirectionDotGradient):
            alphaCubicInter = cubicApproximation(alphaLow, phiCurrentAlphaLow,
                                                 currentSearchDirectionDotGradientAlphaLow, alphaHi, phiCurrentAlphaHi,
                                                 currentSearchDirectionDotGradientAlphaHi)

            if (alphaCubicInter > 0 and alphaCubicInter <= 1):
                alphaJ = alphaCubicInter
                phiCurrentAlphaJ =self.lineSearchEvaluation(alphaJ, params)

        # Bisection interpolation if quadratic goes wrong
        if (alphaJ == 0):
            alphaJ = alphaLow + (alphaHi - alphaLow) / 2
            phiCurrentAlphaJ =self.lineSearchEvaluation(alphaJ, params)

        if ((phiCurrentAlphaJ > phi0 + c1 * alphaJ * initialSearchDirectionDotGradient) or (
                phiCurrentAlphaJ >= phiCurrentAlphaLow)):
            alphaHi = alphaJ
        else:
            # Compute \phi'(\alpha_{j})
            currentSearchDirectionDotGradient = computeDirectionDescent(currNetwork)

            if (abs(currentSearchDirectionDotGradient) <= -c2 * initialSearchDirectionDotGradient):
                return alphaJ

            if (currentSearchDirectionDotGradient * (alphaHi - alphaLow) >= 0):
                alphaHi = alphaLow
            alphaLow = alphaJ

        i = i + 1
    return alphaJ
