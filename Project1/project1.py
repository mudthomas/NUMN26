from assimulo.explicit_ode import Explicit_ODE
from assimulo.solvers.sundials import CVode
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl

import numpy as np


def f(t, y):
    k = 1
    lam = k * (np.sqrt(y[0]**2+y[1]**2) - 1)/np.sqrt(y[0]**2+y[1]**2)
    return np.asarray([y[2], y[3], -y[0]*lam, -y[1]*lam -1])

starting_point = np.array([1,1,1,1])

problem = Explicit_Problem(f, y0=starting_point)
problem.name = "Task 1"

problem_cvode = CVode(problem)
problem_cvode.simulate(10.0, 100)
problem_cvode.plot()
