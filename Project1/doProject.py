from assimulo.solvers.sundials import CVode
import numpy as np
from assimulo.ode import Explicit_Problem

import BDF_FPI as FPI
import BDF_Newton as NEW
import BDF2_CodeByClaus as CLAUS


def doTask1():
    def problem_func(t, y):
        temp = k * ((np.sqrt(y[0]**2+y[1]**2) - 1)/np.sqrt(y[0]**2+y[1]**2))
        return np.asarray([y[2], y[3], -y[0]*temp, -y[1]*temp - 1])
    starting_point = np.array([1, 0, 0, 0])
    problem = Explicit_Problem(problem_func, y0=starting_point)

    for k in [1, 5, 10, 15, 20]:
        problem.name = f"Task 1, k={k}"
        solver = CVode(problem)
        solver.simulate(50)
        solver.plot()  # kwargs at: matplotlib.sourceforge.net/api/pyplot_api.html

def doTask3():
    def problem_func(t, y):
        temp = spring_constant * (1 - 1/np.sqrt(y[0]**2+y[1]**2))
        return np.asarray([y[2], y[3], -y[0]*temp, -y[1]*temp - 1])

    for spring_constant in [5]:
        starting_point = np.array([1, 0, 0, 0])
        problem = Explicit_Problem(problem_func, y0=starting_point)
        t_end = 10

        problem.name = f"Task 3, k={spring_constant}, CVode"
        solver = CVode(problem)
        solver.simulate(t_end)
        solver.plot()

        # problem.name = f"Task 3, k={spring_constant}, Claus Example Code (BDF2, FPI)"
        # solver = CLAUS.BDF_2(problem)
        # solver.simulate(t_end)
        # solver.plot()

        # problem.name = f"Task 3, k={spring_constant}, EE-solver"
        # solver = FPI.EE_solver(problem)
        # solver.simulate(t_end)
        # solver.plot()

        # problem.name = f"Task 3, k={spring_constant}, BDF2-solver, FPI"
        # solver = FPI.BDF2(problem)
        # solver.simulate(t_end)
        # solver.plot()

        # problem.name = f"Task 3, k={spring_constant}, BDF3-solver, FPI"
        # solver = FPI.BDF3(problem)
        # solver.simulate(t_end)
        # solver.plot()

        # problem.name = f"Task 3, k={spring_constant}, BDF4-solver, FPI"
        # solver = FPI.BDF4(problem)
        # solver.simulate(t_end)
        # solver.plot()

        # problem.name = f"Task 3, k={spring_constant}, BDF5-solver, FPI"
        # solver = FPI.BDF5(problem)
        # solver.simulate(t_end)
        # solver.plot()

        # problem.name = f"Task 3, k={spring_constant}, BDF6-solver, FPI"
        # solver = FPI.BDF6(problem)
        # solver.simulate(t_end)
        # solver.plot()

        problem.name = f"Task 3, k={spring_constant}, BDF2-solver, Newton"
        solver = NEW.BDF2_Newton(problem)
        solver.simulate(t_end)
        solver.plot()

        problem.name = f"Task 3, k={spring_constant}, BDF3-solver, Newton"
        solver = NEW.BDF3_Newton(problem)
        solver.simulate(t_end)
        solver.plot()

# doTask1()
doTask3()
