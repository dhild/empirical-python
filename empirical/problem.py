import logging
import numpy as np
from numpy.matlib import repmat
from scipy.linalg import (lu_factor, lu_solve, lstsq)

log = logging.getLogger(__name__)

class BoundaryValueSolution2D:
    def __init__(self, coeffs, bases, domain):
        self.coeffs = coeffs
        self.bases = bases
        self.domain = domain

    def pointEval(self, x):
        k = 1.0 / domain.indexOfRefraction
        values = np.asarray([basis(k, x) for basis in bases])
        return np.sum(np.multiply(values, coeffs))
        
    def __eval__(self, x):
        def func(x):
            return self.pointEval(x)
        X = np.asarray(x)
        return X.unaryExpr(func)

def sqrtQuadratureWeights(domain):
    weights = np.concatenate([dom.boundary.weights() for dom in domains])
    return np.sqrt(weights).reshape((weights.size, 1))

def make_rhs(domain, sqrtWeights, **kwargs):
    bc = domain.appliedBC(**kwargs)
    return np.multiply(bc, sqrtWeights)

def make_A(domain, bases, sqrtWeights, **kwargs):
    N = reduce(lambda x, y: x + y, [b.size() for b in bases])
    M = sqrtWeights.size()
    log.debug('Filling %d x %d basis function matrix', N, M)

    A = np.asmatrix(np.zeros((M, N), dtype='complex'))
    n0 = 0
    for i in range(len(bases)):
        basis = bases[i]
        n1 = n0 + basis.size()
        log.debug('Applying block (%d,%d:%d) basis, total size (%d,%d)', M, n0, n1, M, N)
        Ablock = domain.appliedBasis(basis, **kwargs)
        weights = repmat(sqrtWeights, 1, basis.size())
        np.multiply(Ablock, weights, A[..., n0:n1])
        n0 += n1
    return A

def linsolve(A, rhs):
    if (A.rows() == A.cols()):
        log.info('Solving using LU factorization. Matrix shape: %s', A.shape)
        coeffs = lu_solve(lu_factor(A), rhs)
    else:
        coeffs, res, rank, _ = lstsq(A, rhs)
        log.info('Residues: %s', res)
        log.info('Matrix shape: %s, effective rank: %d', A.shape, rank)

    return np.asmatrix(np.reshape(coeffs, (coeffs.size, 1)))

class BoundaryValueProblem2D:
    bases = []
    def __init__(self, domain):
        self.domain = domain

    def solve(self, **kwargs):
        sqrtWeights = sqrtQuadratureWeights(domain)
        A = make_A(domain, self.bases, sqrtWeights)
        rhs = make_rhs(domain, sqrtWeights, **kwargs)
        coeffs = linsolve(A, rhs)
        return BoundaryValueSolution2D(coeffs, self.bases, self.domain)
