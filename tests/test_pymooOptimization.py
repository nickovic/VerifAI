import os
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.static import StaticProblem
from pymoo.core.evaluator import Evaluator
from verifai.features import *
from verifai.samplers import *
from dotmap import DotMap
from numpy import random as rdm

import pytest
from tests.utils import sampleWithFeedback, checkSaveRestore


def test_pymooOptimization():
    carDomain = Struct({
        'position': Box([-10,10], [-10,10], [0,1]),
        'heading': Box([0, math.pi]),
    })

    space = FeatureSpace({
        'cars': Feature(Array(carDomain, [2]))
    })

    def f(sample):
        sample = sample.cars[0].heading[0]
        return abs(sample - 0.75)

    params = DotMap()
    params.n_var=8
    sampler = FeatureSampler.pymooSamplerFor(space, params)

    sampleWithFeedback(sampler, 10, f)

def test_pymooOptimization_mo():
    carDomain = Struct({
        'position': Box([-10,10], [-10,10], [0,1]),
        'heading': Box([0, math.pi]),
    })

    space = FeatureSpace({
        'cars': Feature(Array(carDomain, [2]))
    })

    def f(sample):
        sample = sample.cars[0].heading[0]
        return abs(sample - 0.75)

    def g(sample):
        sample = sample.cars[0].heading[0]
        return abs(sample - 0.8)

    params = DotMap()
    params.n_var = 8
    params.n_obj = 2
    sampler = FeatureSampler.pymooSamplerFor(space, params)

    feedback = None
    samples = []
    for i in range(10):
        sample = sampler.nextSample(feedback)
        feedback = [f(sample), g(sample)]
        print(f'Sample #{i}:')
        print(sample)
        samples.append(sample)


def test_save_restore(tmpdir):
    space = FeatureSpace({
        'a': Feature(DiscreteBox([0, 12])),
        'b': Feature(Box((0, 1)), lengthDomain=DiscreteBox((1, 2)))
    })
    params = DotMap()
    params.n_var = 2
    params.n_obj = 1
    sampler = FeatureSampler.pymooSamplerFor(space, params)

    path = os.path.join(tmpdir, 'blah.dat')
    feedback = None
    for i in range(10):
        sampler.saveToFile(path)
        sample1 = sampler.nextSample(feedback)
        sample2 = sampler.nextSample(-1.1)
        sampler = FeatureSampler.restoreFromFile(path)
        sample1b = sampler.nextSample(feedback)
        sample2b = sampler.nextSample(-1.1)
        assert sample1 != sample2
        assert sample1 == sample1b
        assert sample2 == sample2b
        sampler.saveToFile(path)
        sample3 = sampler.nextSample(1.1)
        sampler = FeatureSampler.restoreFromFile(path)
        sample3b = sampler.nextSample(1.1)
        assert sample3 not in (sample1, sample2)
        assert sample3 == sample3b
        feedback = 1.1


def test_direct_vs_verifai():
    def fun(x): return ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) *
                     x[0] ** 2 + x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)


    # Direct Pymoo approach
    rdm.seed(0)
    random.seed(0)

    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])

    p = {'n_var': 2, 'xl': lb, 'xu': ub}
    problem = Problem(**p)
    termination = NoTermination()
    algorithm = NSGA2(pop_size=1)
    algorithm.setup(problem, termination=termination)

    X = []
    F = []
    for i in range(6):
        pop = algorithm.ask()
        x = pop.get("X")
        f = fun(x[0])
        X.append(x[0].tolist())
        F.append(f)
        static = StaticProblem(problem, F=[f])
        Evaluator().eval(static, pop)
        algorithm.tell(infills=pop)

    # VerifAI approach
    rdm.seed(0)
    random.seed(0)
    space = FeatureSpace({
        'a': Feature(Box([-2.0, 2.0])),
        'b': Feature(Box([-1.0, 1.0]))
    })

    sampler = FeatureSampler.pymooSamplerFor(space, p)

    X_prime = []
    F_prime = []
    for i in range(6):
        if i == 0:
            x_prime = sampler.nextSample(int(1))
            X_prime.append([x_prime[0][0], x_prime[1][0]])
        else:
            f_prime = fun([x_prime[0][0], x_prime[1][0]])
            x_prime = sampler.nextSample(feedback=f_prime)
            X_prime.append([x_prime[0][0], x_prime[1][0]])
            F_prime.append(f_prime)

    assert X == X_prime
    assert F[0:len(F_prime)] == F_prime