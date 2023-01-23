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
        self.maxit = 10000
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
        t, y = self.BDF2step(t, y_list[-1], y_list[-2], h)
        t_list.append(t)
        y_list.append(y)
        for i in range(2, self.maxsteps):
            if t >= tf:
                break
            # BDF3
            t, y = self.BDF3step(t, y_list[-1], y_list[-2], y_list[-3], h)
            t_list.append(t)
            y_list.append(y)
        return ID_PY_OK, t_list, y_list

    def EEstep(self, t_n, y_n, h):
        t_np1 = t_n + h
        y_np1 = y_n + h*self.problem.rhs(t_n, y_n)
        return t_np1, y_np1

    def BDF2step(self, t_n, y_n, y_nm1, h):
        t_np1 = t_n + h
        y_np1 = y_n
        for i in range(self.maxit):
            y_np1_old = y_np1
            temp_func = lambda y: (4*y_n - 1*y_nm1 + 2*h*self.problem.rhs(t_np1, y))/3 -y
            y_np1 = fsolve(temp_func, y_n)
            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception

    def BDF3step(self, t_n, y_n, y_nm1, y_nm2, h):
        t_np1 = t_n + h
        y_np1 = y_n
        for i in range(self.maxit):
            y_np1_old = y_np1
            temp_func = lambda y: (18*y_n - 9*y_nm1 + 2*y_nm2 + 6*h*self.problem.rhs(t_np1, y))/11 - y
            y_np1 = fsolve(temp_func, y_n)
            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception

    def BDFstep_general(self, t_n, Y, multipliers, divisor, h):
        t_np1 = t_n + h
        y_np1 = Y[0]
        for i in range(self.maxit):
            y_np1_old = y_np1

            clump = 0
            for i in range(len(Y)-1):
                clump += multipliers[i] * Y[i]

            temp_func = lambda y: (clump+multipliers[-1]*self.problem.rhs(t_np1, y))/divisor - y
            y_np1 = fsolve(temp_func, Y[0])

            y_np1 /= divisor

            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)


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
        t, y = self.BDF2step(t, y_list[-1], y_list[-2], h)
        t_list.append(t)
        y_list.append(y)
        # one step BDF3
        t, y = self.BDF3step(t, y_list[-1], y_list[-2], y_list[-3], h)
        t_list.append(t)
        y_list.append(y)
        for i in range(3, self.maxsteps):
            if t >= tf:
                break
            # BDF4
            t, y = self.BDF4step(t, y_list[-1], y_list[-2], y_list[-3], y_list[-4], h)
            t_list.append(t)
            y_list.append(y)
        return ID_PY_OK, t_list, y_list

    def EEstep(self, t_n, y_n, h):
        t_np1 = t_n + h
        y_np1 = y_n + h*self.problem.rhs(t_n, y_n)
        return t_np1, y_np1

    def BDF2step(self, t_n, y_n, y_nm1, h):
        t_np1 = t_n + h
        y_np1 = y_n
        for i in range(self.maxit):
            y_np1_old = y_np1
            temp_func = lambda y: (4*y_n - 1*y_nm1 + 2*h*self.problem.rhs(t_np1, y))/3 -y
            y_np1 = fsolve(temp_func, y_n)
            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception

    def BDF3step(self, t_n, y_n, y_nm1, y_nm2, h):
        t_np1 = t_n + h
        y_np1 = y_n
        for i in range(self.maxit):
            y_np1_old = y_np1
            temp_func = lambda y: (18*y_n - 9*y_nm1 + 2*y_nm2 + 6*h*self.problem.rhs(t_np1, y))/11 - y
            y_np1 = fsolve(temp_func, y_n)
            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception

    def BDF4step(self, t_n, y_n, y_nm1, y_nm2, y_nm3, h):
        t_np1 = t_n + h
        y_np1 = y_n
        for i in range(self.maxit):
            y_np1_old = y_np1
            temp_func = lambda y: (48*y_n - 36*y_nm1 + 16*y_nm2 - 3*y_nm3 + 12*h*self.problem.rhs(t_np1, y))/25 - y
            y_np1 = fsolve(temp_func, y_n)
            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)

    def BDFstep_general(self, t_n, Y, multipliers, divisor, h):
        t_np1 = t_n + h
        y_np1 = Y[0]
        for i in range(self.maxit):
            y_np1_old = y_np1
            clump = 0
            for i in range(len(Y)-1):
                clump += multipliers[i] * Y[i]
            temp_func = lambda y: (clump + multipliers[-1]*self.problem.rhs(t_np1, y))/divisor -y
            y_np1 = fsolve(temp_func, Y[0])

            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)


starting_point = np.array([1, 0, 0, -1])
problem = Explicit_Problem(f, y0=starting_point)

for k in [0, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
    problem.name = f"Task 1, k={k}"
    solver = BDF_4(problem)
    solver.simulate(10.0, 100)
    solver.plot()