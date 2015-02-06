import unittest
from unittest.mock import *
import numpy as np
import math
from empirical.basis import *

e = math.exp(1)

class MFSBasis2DTest(unittest.TestCase):
    def test_mfsbasis_function_k0(self):
        val = mfsBasis(0, 1)
        self.assertAlmostEqual(val, 0)
        val = mfsBasis(0, -1)
        self.assertAlmostEqual(val, 0)
        val = mfsBasis(0, e)
        self.assertAlmostEqual(val, -0.5 / math.pi)

    def test_mfsbasis_function_k(self):
        val = mfsBasis(1, 1)
        self.assertAlmostEqual(val, 0.25j * hankel(0, 1))
        val = mfsBasis(1, -1)
        self.assertAlmostEqual(val, 0.25j * hankel(0, 1))
        val = mfsBasis(15, e)
        self.assertAlmostEqual(val, 0.25j * hankel(0, 15 * e))

    def test_mfsbasisderiv_function_k0(self):
        val = mfsBasisDerivative(0, 1)
        self.assertAlmostEqual(val, -0.5 / math.pi)
        val = mfsBasisDerivative(0, -1)
        self.assertAlmostEqual(val, -0.5 / math.pi)
        val = mfsBasisDerivative(0, e)
        self.assertAlmostEqual(val, -0.5 / (e * math.pi))

    def test_mfsbasisderiv_function_k(self):
        val = mfsBasisDerivative(1, 1)
        self.assertAlmostEqual(val, (-1 + 0.25j) * hankel(1, 1))
        val = mfsBasisDerivative(1, -1)
        self.assertAlmostEqual(val, (-1 + 0.25j) * hankel(1, 1))
        val = mfsBasisDerivative(15, e)
        self.assertAlmostEqual(val, (-15 + 0.25j) * hankel(1, e))

    def test_init(self):
        bnd = Mock()
        bas = MFSBasis2D(bnd)
        self.assertIs(bas.boundary, bnd)

    def test_size(self):
        bnd = Mock()
        bnd.N = 523
        bas = MFSBasis2D(bnd)
        size = bas.size()
        self.assertEqual(size, 523)

    def test_eval_x(self):
        bnd = Mock()
        bnd.points = Mock()
        bas = MFSBasis2D(bnd)
        val = bas(1.025, -2.5, 5)
        self.assertEqual(val, mfsBasis(1.025, -7.5))
        self.assertFalse(bnd.points.called)
