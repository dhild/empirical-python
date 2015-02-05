import logging

log = logging.getLogger(__name__)

class BaseDomain2D:
    boundaryCondition = 0
    _indexOfRefraction = 1.0

    def __init__(self, boundary, flipNormals=False):
        self.boundary = boundary
        self.flipNormals = flipNormals

    @property
    def wavenumber(self):
        return 1.0 / _indexOfRefraction

    @wavenumber.setter
    def wavenumber(self, value):
        _indexOfRefraction = 1.0 / value

    @property
    def indexOfRefraction(self):
        return _indexOfRefraction

    @indexOfRefraction.setter
    def indexOfRefraction(self, value):
        _indexOfRefraction = value

    def appliedBC(self, **kwargs):
        if (callable(self.boundaryCondition)):
            a = self.boundaryCondition(self.boundary, **kwargs)
            return np.reshape(a, (self.boundary.N, 1))
        return boundaryCondition

    def appliedBasis(self, basis, **kwargs):
        k = self.wavenumber
        x = self.boundary.points()
        return basis(k, x)

class ExteriorDomain2D(BaseDomain2D):
    def __init__(self, boundary, flipNormals=False, indexOfRefraction=1.0):
        super().__init__(boundary, flipNormals, indexOfRefraction)

    def isExterior(self):
        return True

class InteriorDomain2D(BaseDomain2D):
    def __init__(self, boundary, flipNormals=False, indexOfRefraction=1.0):
        super().__init__(boundary, flipNormals, indexOfRefraction)

    def isExterior(self):
        return False