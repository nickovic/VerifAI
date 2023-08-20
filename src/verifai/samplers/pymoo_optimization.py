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
        # we do some sanity checks on rho
        # rho can be:
        # - None (not initialized yet)
        # - int(1) (sample rejected, hence rho not computed)
        # - float (number of objectives must be one)
        # - list of floats (the length of the list corresponds to the number of objectives

        if self.rho is not None:
            is_float = isinstance(self.rho, float)
            is_int = isinstance(self.rho, int)
            is_list = isinstance(self.rho, list)
            assert (is_float or is_list or is_int)
            if is_float:
                assert self.problem.n_obj == 1
                self.rho = [self.rho]
            elif is_list:
                assert (all(isinstance(x, float) for x in self.rho))
                assert (len(self.rho) == self.problem.n_obj)

            if not isinstance(self.rho, int):
                static = StaticProblem(self.problem, F=self.rho)
                Evaluator().eval(static, self.pop)
                self.algorithm.tell(infills=self.pop)

        if not isinstance(self.rho, int):
            self.pop = self.algorithm.ask()

        x = self.pop.get("X")

        return tuple(x[0]), None

    def updateVector(self, vector, info, rho):
        self.rho = rho
