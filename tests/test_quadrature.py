import unittest
import numpy as np
from empirical.quadrature import *

def function(x):
    return x * np.sin(30 * x) + np.cos(5 * x)

def function_integral(x):
    return (-(1 / 30) * x * np.cos(30 * x) +
             (1 / 5) * np.sin(5 * x) +
             (1 / 900) * np.sin(30 * x))

integration_value = function_integral(1) - function_integral(-1)

class QuadratureTest(unittest.TestCase):
    def test_linear(self):
        N = 250
        x, w = linear(N)
        self.assertEqual(x.shape, (N,))
        self.assertEqual(w.shape, (N,))
        self.assertEqual(x[0], -1)
        self.assertEqual(x[-1], 1)
        integrate = np.sum(x * w)
        self.assertAlmostEqual(integrate, 0, places=10)
        # no use testing sinusoid integration, it'll be widely off

    def test_pt(self):
        N = 250
        x, w = periodic_trapezoid(N)
        self.assertEqual(x.shape, (N,))
        self.assertEqual(w.shape, (N,))
        self.assertNotIn(-1, x)
        self.assertNotIn(1, x)
        integrate = np.sum(x * w)
        self.assertAlmostEqual(integrate, 0, places=10)
        integrate = np.sum(function(x) * w)
        self.assertAlmostEqual(integrate, integration_value, places=4)

    def test_t(self):
        N = 251
        x, w = trapezoid(N)
        self.assertEqual(x.shape, (N,))
        self.assertEqual(w.shape, (N,))
        self.assertEqual(x[0], -1)
        self.assertEqual(x[-1], 1)
        integrate = np.sum(x * w)
        self.assertAlmostEqual(integrate, 0, places=10)
        integrate = np.sum(function(x) * w)
        self.assertAlmostEqual(integrate, integration_value, places=2)

    def test_lgl(self):
        N = 250
        x, w = legendre_gauss_lobatto(N)
        self.assertEqual(x.shape, (N,))
        self.assertEqual(w.shape, (N,))
        integrate = np.sum(x * w)
        self.assertAlmostEqual(integrate, 0, places=10)
        integrate = np.sum(function(x) * w)
        self.assertAlmostEqual(integrate, integration_value, places=10)