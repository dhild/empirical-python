import logging
from math import pi
import numpy as np

log = logging.getLogger(__name__)

def linear(N):
    return (np.linspace(-1, 1, N), np.ones(N))

def trapezoid(N):
    points = np.linspace(-1.0, 1.0, N)
    w = 1.0 / float(N)
    weights = np.ones(N) * 2.0 * w
    weights[0] = weights[-1] = w
    return (points, weights)

def periodic_trapezoid(N):
    shift = 1.0 / float(N)
    points = np.linspace(-1 + shift, 1 - shift, N)
    weights = np.ones(N) * (2.0 / float(N))
    return (points, weights)

def legendre_gauss_lobatto(N):
    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.linspace(0, pi, N))
    # The Legendre Vandermonde Matrix
    P = np.zeros((N, N))
    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.
    xold = 2
    while np.amax(np.abs(x - xold)) > (1e-15):
        xold = x
        P[:, 0] = 1
        P[:, 1] = x
        for k in range(2, N):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] -
                       (k - 1) * P[:, k - 2]) / k
        x = xold - (x * P[:, N - 1] - P[:, N - 2]) / (N * P[:, N - 1])

    w = 2.0 / ((N - 1) * N * np.square(P[:, N - 1]))

    return (np.array(x[::-1]), np.array(w[::-1]))
