from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem, ElementwiseEvaluationFunction, LoopedElementwiseEvaluation
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from verifai.samplers.domain_sampler import BoxSampler
import numpy as np


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

        self.pop = None
        self.current_rho = None
        self.current_x = None
        self.rho_array = []
        self.x_array = []



        dim = domain.flattenedDimension
        params["n_var"] = dim
        params["xl"] = zeros(dim)
        params["xu"] = ones(dim)

        self.problem = Problem(**params)

        if 'algorithm' not in params:
            params['algorithm'] = NSGA2(params['pop_size'])
        self.algorithm = params['algorithm']

        if hasattr(self.algorithm, 'pop_size'):
            params['pop_size'] = self.algorithm.pop_size
        elif 'pop_size' not in params:
            params['pop_size'] = 1
            self.algorithm = params['algorithm']


        termination = NoTermination()
        self.algorithm.setup(self.problem, termination=termination)

        self.pop_size = params['pop_size']
        self.counter = 0

    def getVector(self):
        # When pop_size outputs are collected, we evaluate them
        # we ignore iterations when the sample is rejected (rho = int(1))
        if len(self.rho_array) == self.pop_size and self.current_rho != int(1):
            static = StaticProblem(self.problem, F=self.rho_array)
            Evaluator().eval(static, self.pop)
            self.algorithm.tell(infills=self.pop)
            self.rho_array = list()

        # We check if we need to generate new array of samples
        # or we just use samples from the generation that was
        # previously generated
        if not self.x_array:
            self.pop = self.algorithm.ask()
            self.x_array = self.pop.get("X").tolist()

        # we do some sanity checks on rho
        # current rho can be:
        # - None (not initialized yet)
        # - int(1) (sample rejected, hence rho not computed)
        # - float (number of objectives must be one)
        # - list of floats (the length of the list corresponds to the number of objectives)

        if self.current_rho is not None:
            is_float = isinstance(self.current_rho, float)
            is_int = isinstance(self.current_rho, int)
            is_list = isinstance(self.current_rho, list)
            assert (is_float or is_list or is_int)
            if is_float:
                assert self.problem.n_obj == 1
                self.current_rho = [self.current_rho]
            elif is_list:
                assert (all(isinstance(x, float) for x in self.current_rho))
                assert (len(self.current_rho) == self.problem.n_obj)

            if not isinstance(self.current_rho, int):
                self.rho_array.append(self.current_rho)


        # Do not remove a sample if it was rejected
        if self.current_rho == int(1):
            out = tuple(self.x_array[0])
        else:
            out = tuple(self.x_array.pop(0))


        return out, None

    def updateVector(self, vector, info, rho):
        self.current_rho = rho
