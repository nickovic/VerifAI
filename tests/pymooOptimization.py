import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

problem = Problem(n_var=30, n_obj=2, n_constr=0, xl=np.zeros(30), xu=np.ones(30))

# create the algorithm object
algorithm = NSGA2(pop_size=1)

# let the algorithm object never terminate and let the loop control it
termination = NoTermination()

# create an algorithm object that never terminates
algorithm.setup(problem, termination=termination)

# fix the random seed manually
np.random.seed(1)

# until the algorithm has no terminated
for n_gen in range(10):
    # ask the algorithm for the next solution to be evaluated
    pop = algorithm.ask()

    # get the design space values of the algorithm
    X = pop.get("X")

    # implement your evluation. here ZDT1
    f1 = X[:, 0]
    v = 1 + 9.0 / (problem.n_var - 1) * np.sum(X[:, 1:], axis=1)
    f2 = v * (1 - np.power((f1 / v), 0.5))
    F = np.column_stack([f1, f2])

    static = StaticProblem(problem, F=F)
    Evaluator().eval(static, pop)

    # returned the evaluated individuals which have been evaluated or even modified
    algorithm.tell(infills=pop)

    # do same more things, printing, logging, storing or even modifying the algorithm object
    print(str(algorithm.n_gen) + ' ' + str(X[0]) + " " + str(F[0]))

# obtain the result objective from the algorithm
res = algorithm.result()

# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())