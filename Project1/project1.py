from assimulo.solvers.sundials import CVode
from assimulo.ode import Explicit_Problem
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *

import numpy as np
from scipy.optimize import fsolve


def f(t, y):
    # k=1
    lam = k * (np.sqrt(y[0]**2+y[1]**2) - 1)/np.sqrt(y[0]**2+y[1]**2)
    return np.asarray([y[2], y[3], -y[0]*lam, -y[1]*lam - 1])

# starting_point = np.array([1, 0, 0, -1])
# problem = Explicit_Problem(f, y0=starting_point)

# for k in [0, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
#     problem.name = f"Task 1, k={k}"
#     solver = CVode(problem)
#     solver.simulate(10.0, 100)
#     solver.plot()


class BDF_general(Explicit_ODE):
    def __init__(self, problem, degree):
        Explicit_ODE.__init__(self, problem)
        self.maxsteps = 500
        self.maxit = 100
        self.tol = 1.e-8
        self.degree = degree

        #Solver options
        self.options["h"] = 0.01
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _set_h(self,h):
            self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    def integrate(self, t, y, tf, opts):
        h = self.options["h"]
        h = min(h, tf-t)  # Claus uses abs(tf-t), I'd rather have a check for tf>t.

        t_list = [t]
        y_list = [y]
        # one step EE
        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t, y, h)
        t_list.append(t)
        y_list.append(y)

        for i in range(1, self.maxsteps):
            if t >= tf:
                break
            # BDF
            self.statistics["nsteps"] += 1
            try:
                t, y = self.BDFstep_general(t, y_list[-self.degree:][::-1], h)
            except:
                t, y = self.BDFstep_general(t, y_list[::-1], h)
            t_list.append(t)
            y_list.append(y)
            h = min(h, np.abs(tf-t))

        return ID_PY_OK, t_list, y_list

    def EEstep(self, t_n, y_n, h):
        self.statistics["nfcns"] += 1
        return t_n + h, h*self.problem.rhs(t_n, y_n)

    def BDFstep_general(self, t_n, Y, h):
        coeffs=[[1, -1, -1],
                [3, -4, 1, -2],
                [11, -18, 9, -2, -6],
                [25, -48, 36, -16, 3, -12],
                [137, -300, 300, -22, 75, -12, -60],
                [147, -360, 450, -400, 225, -72, 10, -60]]
        return self.BDFstep_general_internal(t_n, Y, coeffs[len(Y)-1], h)

    def BDFstep_general_internal(self, t_n, Y, coeffs, h):
        t_np1 = t_n + h
        y_np1 = Y[0]

        clump = 0
        for i in range(len(Y)):
            clump += coeffs[i+1] * Y[i]

        for i in range(self.maxit):
            self.statistics["nfcns"] += 1

            y_np1_old = y_np1
            # FPI
            # y_np1 = -(clump + coeffs[-1]*h*self.problem.rhs(t_np1, y_np1_old))/coeffs[0]

            # With fsolve
            temp_func = lambda y: coeffs[0]*y + (clump + coeffs[-1]*h*self.problem.rhs(t_np1, y))
            y_np1 = fsolve(temp_func, y_np1_old)

            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")

class BDF_3(BDF_general):
    def __init__(self, problem):
        BDF_general.__init__(self, problem, 3)

class BDF_4(BDF_general):
    def __init__(self, problem):
        BDF_general.__init__(self, problem, 4)

starting_point = np.array([1, 0, 0, -1])
problem = Explicit_Problem(f, y0=starting_point)

for k in [0.1, 1, 10, 100, 1000]:
    problem.name = f"Task 1, k={k}"
    solver = BDF_4(problem)
    solver.simulate(10.0, 100)
    solver.plot()
