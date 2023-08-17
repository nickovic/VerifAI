"""Pymoo Optimization sampler : Defined only for continuous domains.
For discrete inputs define another sampler"""

from glis.solvers import GLIS
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem, ElementwiseEvaluationFunction, LoopedElementwiseEvaluation
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from verifai.samplers.domain_sampler import BoxSampler


class PymooSampler(BoxSampler):
    '''
    Integrates the Pymoo sampler with VerifAI
    ---
    Parameters:
        domain : FeatureSpace
        params : DotMap
    ---
    Note: see the definition of the Pymoo class for the available parameters or the Pymoo documentation
    https://pymoo.org/misc/index.html
    '''

    def __init__(self, domain, params):
        super().__init__(domain)
        from numpy import zeros, ones

        self.rho = None
        self.pop = None

        dim = domain.flattenedDimension
        params["n_var"] = dim
        params["xl"] = zeros(dim)
        params["xu"] = ones(dim)


        self.problem = Problem(**params)

        if 'algorithm' not in params:
            params['algorithm'] = NSGA2(pop_size=1)

        self.algorithm = params['algorithm']
        termination = NoTermination()
        self.algorithm.setup(self.problem, termination=termination)

    def getVector(self):
        if self.rho is not None:
            if isinstance(self.rho, float):
                self.rho = [self.rho]
            static = StaticProblem(self.problem, F=self.rho)
            Evaluator().eval(static, self.pop)
            self.algorithm.tell(infills=self.pop)

        if self.rho != int(1):
            self.pop = self.algorithm.ask()

        x = self.pop.get("X")

        return tuple(x[0]), None

    def updateVector(self, vector, info, rho):
        self.rho = rho
