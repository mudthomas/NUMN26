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


class BDF_3(Explicit_ODE):
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)
        self.maxsteps = 500
        self.maxit = 100
        self.tol = 1.e-8

    def integrate(self, t, y, tf, opts):
        t0 = t
        y0 = y
        h = 0.01
        t_list = [t0]
        y_list = [y0]
        # one step EE
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        # one step BDF2
        t, y = self.BDFstep_general(t, [y_list[-1], y_list[-2]], h)
        t_list.append(t)
        y_list.append(y)
        for i in range(2, self.maxsteps):
            if t >= tf:
                break
            # BDF3
            t, y = self.BDFstep_general(t, [y_list[-1], y_list[-2], y_list[-3]], h)
            t_list.append(t)
            y_list.append(y)
        return ID_PY_OK, t_list, y_list

    def EEstep(self, t_n, y_n, h):
        t_np1 = t_n + h
        y_np1 = y_n + h*self.problem.rhs(t_n, y_n)
        return t_np1, y_np1

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
        for i in range(self.maxit):
            y_np1_old = y_np1
            clump = 0
            for i in range(len(Y)):
                clump += coeffs[i+1] * Y[i]

            # With fsolve
            # temp_func = lambda y: coeffs[0]*y + (clump + coeffs[-1]*h*self.problem.rhs(t_np1, y))
            # y_np1 = fsolve(temp_func, y_np1_old)

            # FPI
            y_np1 = -(clump + coeffs[-1]*h*self.problem.rhs(t_np1, y_np1))/coeffs[0]

            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


class BDF_4(Explicit_ODE):
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)
        self.maxsteps = 500
        self.maxit = 100
        self.tol = 1.e-8

    def integrate(self, t, y, tf, opts):
        t0 = t
        y0 = y
        h = 0.01
        t_list = [t0]
        y_list = [y0]
        # one step EE
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        # one step BDF2
        t, y = self.BDF2step(t, [y_list[-1], y_list[-2]], h)
        t_list.append(t)
        y_list.append(y)
        # one step BDF3
        t, y = self.BDF3step(t, [y_list[-1], y_list[-2], y_list[-3]], h)
        t_list.append(t)
        y_list.append(y)
        for i in range(3, self.maxsteps):
            if t >= tf:
                break
            t, y = self.BDF4step(t, [y_list[-1], y_list[-2], y_list[-3], y_list[-4]], h)
            t_list.append(t)
            y_list.append(y)
        return ID_PY_OK, t_list, y_list

    def EEstep(self, t_n, y_n, h):
        t_np1 = t_n + h
        y_np1 = y_n + h*self.problem.rhs(t_n, y_n)
        return t_np1, y_np1

    def BDF2step(self, t_n, Y, h):
        return self.BDFstep_general(t_n, Y, [3, -4, 1, -2], h)

    def BDF3step(self, t_n, Y, h):
        return self.BDFstep_general(t_n, Y, [11, -18, 9, -2, -6], h)

    def BDF4step(self, t_n, Y, h):
        return self.BDFstep_general(t_n, Y, [25, -48, 36, -16, 3, -12], h)

    def BDFstep_general(self, t_n, Y, coeffs, h):
        t_np1 = t_n + h
        y_np1 = Y[0]
        for i in range(self.maxit):
            y_np1_old = y_np1
            clump = 0
            for i in range(len(Y)):
                clump += coeffs[i+1] * Y[i]

            # With fsolve
            # temp_func = lambda y: coeffs[0]*y + (clump + coeffs[-1]*h*self.problem.rhs(t_np1, y))
            # y_np1 = fsolve(temp_func, y_np1_old)

            # FPI
            y_np1 = -(clump + coeffs[-1]*h*self.problem.rhs(t_np1, y_np1))/coeffs[0]

            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")

starting_point = np.array([1, 0, 0, -1])
problem = Explicit_Problem(f, y0=starting_point)

for k in [0.1, 1, 10, 100, 1000]:
    problem.name = f"Task 1, k={k}"
    solver = BDF_4(problem)
    solver.simulate(10.0, 100)
    solver.plot()
