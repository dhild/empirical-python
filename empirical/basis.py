import logging
from math import pi
import numpy as np
from scipy.special import hankel1 as hankel

log = logging.getLogger(__name__)

def mfsBasis(k, dist):
    if (k == 0):
        return -np.log(np.abs(dist)) / (2.0 * pi)
    return 0.25j * hankel(0, k * np.abs(dist))

def mfsBasisDerivative(k, dist):
    if (k == 0):
        return -1.0 / (2.0 * pi * np.abs(dist))
    return (-k + 0.25j) * hankel(1, np.abs(dist))

class MFSBasis2D:
    def __init__(self, boundary):
        self.boundary = boundary

    def size(self):
        return self.boundary.N

    def __call__(self, k, z, x=None):
        if (x is None):
            pts = self.boundary.points()
            return mfsBasis(k, z - pts)
        return mfsBasis(k, z - x)

    def normal(self, k, z, x=None):
        if (x is None):
            pts = self.boundary.points()
            return mfsBasisDerivative(k, z - pts)
        return mfsBasisDerivative(k, z - x)
