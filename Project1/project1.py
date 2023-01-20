from assimulo.solvers.sundials import CVode
from assimulo.ode import Explicit_Problem
import numpy as np


def f(t, y):
    # k=1
    lam = k * (np.sqrt(y[0]**2+y[1]**2) - 1)/np.sqrt(y[0]**2+y[1]**2)
    return np.asarray([y[2], y[3], -y[0]*lam, -y[1]*lam - 1])


starting_point = np.array([1, 1, 1, 1])
problem = Explicit_Problem(f, y0=starting_point)

for k in [0, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
    problem.name = f"Task 1, k={k}"
    problem_cvode = CVode(problem)
    problem_cvode.simulate(10.0, 100)
    problem_cvode.plot()
