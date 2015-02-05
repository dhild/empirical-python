import logging
from math import pi
import numpy as np

log = logging.getLogger(__name__)

class Boundary2D:
    def __init__(self, z, zPrime, quadrature, N = 50):
        self.z = z
        self.zPrime = zPrime
        self.quadrature = quadrature
        self.N = N

    def points(self):
        x, w = quadrature(N)
        return self.z(x)

    def pointDerivatives(self):
        x, w = quadrature(N)
        return self.zPrime(N)

    def weights(self):
        x, w = quadrature(N)
        return w

    def normals(self):
        x, w = quadrature(N)
        zp = self.zPrime(N)
        mag = np.abs(zp)
        return (zp.imag / mag) + (1j) * (-zp.real / mag)

class ArcSegment2D(Boundary2D):
    def __init__(self, center, R, t0, t1, N):
        super().__init__(z=self.points, zPrime=self.pointPrimes, N)
        self.center = center
        self.radius = R
        self.t_scale = (t1 - t0) / 2.0
        self.t_offset = (t1 + t0) / 2.0

    def points(self, x):
        e = np.exp(1j * ((x * self.t_scale) + self.t_offset))
        return self.center + self.radius * e

    def pointPrimes(self, x):
        e = np.exp(1j * ((x * self.t_scale) + self.t_offset))
        return e * 1j * self.radius * self.t_scale

class ComplexFunctionSegment2D(Boundary2D):
    def __init__(self, z, zDeriv, N, offset = 0, scale = 0):
        super().__init__(z=self.points, zPrime=self.pointPrimes, N)
        self.zComplex = z
        self.zDeriv = zDeriv
        self.offset = offset
        self.scale = scale

    def points(self, x):
        arg = self.scale * (x + self.offset)
        return self.zComplex(arg)

    def pointPrimes(self, x):
        arg = self.scale * (x + self.offset)
        return self.zDeriv(arg)

class RadialSegment2D(Boundary2D):
    def __init__(self, radius, radiusDeriv, N):
        super().__init__(z=self.points, zPrime=self.pointPrimes, N)
        self.radius = radius
        self.radiusDeriv = radiusDeriv

    def points(self, x):
        angle = pi * x
        return self.radius(angle) * np.exp(1j * angle)

    def pointPrimes(self, x):
        angle = pi * x
        e = np.exp(1j * angle)
        r = self.radius(angle)
        rp = self.radiusDeriv(angle)
        return PI * ((1j * r * e) + (rp * e))
