"""Pymoo Optimization sampler : Defined only for continuous domains.
For discrete inputs define another sampler"""

from glis.solvers import GLIS
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from verifai.samplers.domain_sampler import BoxSampler


class VerifaiProblem(Problem):
    def __init__(self,
                 n_var=-1,
                 n_obj=1,
                 n_ieq_constr=0,
                 n_eq_constr=0,
                 xl=None,
                 xu=None,
                 vtype=None,
                 vars=None,
                 elementwise=False,
                 elementwise_func=ElementwiseEvaluationFunction,
                 elementwise_runner=LoopedElementwiseEvaluation(),
                 replace_nan_values_by=None,
                 exclude_from_serialization=None,
                 callback=None,
                 strict=True,
                 **kwargs):
        self.out = None
        super().__init__(n_var, n_obj, n_ieq_constr, n_eq_constr, xl, xu, vtype, vars, elementwise,
                         elementwise_func, elementwise_runner, replace_nan_values_by, exclude_from_serialization,
                         callback, strict, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        if 'result' not in kwargs:
            raise Exception('Pymoo Sampler: the output of simulation evaluation '
                            'must be passed as result argument')
        out["F"] = kwargs['result']

    def update(self, out):
        self.out = out


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

        self.problem = Problem(**params)

        if 'algorithm' not in params:
            params['algorithm'] = NSGA2(pop_size=1)

        self.algorithm = params['algorithm']
        termination = NoTermination()
        self.algorithm.setup(self.problem, termination=termination)

    def getVector(self):
        if self.rho is not None:
            static = StaticProblem(self.problem, F=[self.rho])
            Evaluator().eval(static, self.pop)
            self.algorithm.tell(infills=self.pop)

        if self.rho != int(1):
            self.pop = self.algorithm.ask()
            x = self.pop.get("X")

        return tuple(x), None

    def updateVector(self, vector, info, rho):
        self.rho = rho
