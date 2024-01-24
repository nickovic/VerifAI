
from glis.solvers import GLIS
from verifai.samplers.domain_sampler import BoxSampler

class RepeatSampler(BoxSampler):
    def __init__(self, domain, params):
        super().__init__(domain)
        from numpy import zeros, ones

        self.rho = None

        dim = domain.flattenedDimension
        self.lb = zeros(dim)
        self.ub = ones(dim)
        self.sampler = GLIS(bounds=(self.lb, self.ub), **params)
        self.x = self.sampler.initialize()

    def getVector(self):
        return tuple(self.x), None

    def updateVector(self, vector, info, rho):
        self.rho = rho