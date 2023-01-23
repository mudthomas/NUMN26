from assimulo.solvers.sundials import CVode
from assimulo.ode import Explicit_Problem
from assimulo.explicit_ode import Explicit_ODE

import numpy as np


def f(t, y):
    # k=1
    lam = k * (np.sqrt(y[0]**2+y[1]**2) - 1)/np.sqrt(y[0]**2+y[1]**2)
    return np.asarray([y[2], y[3], -y[0]*lam, -y[1]*lam - 1])

starting_point = np.array([1, 0, 0, -1])
problem = Explicit_Problem(f, y0=starting_point)

for k in [0, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
    problem.name = f"Task 1, k={k}"
    solver = CVode(problem)
    solver.simulate(10.0, 100)
    solver.plot()


class BDF_3(Explicit_ODE):
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)
        self.maxsteps = 500
        self.maxit = 100
        self.tol = 1.e-8

    def integrate(self, t0, tf, y0):
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
        pass

    def EEstep(self, t_n, y_n, h):
        t_np1 = t_n + h
        y_np1 = y_n + h*self.problem.rhs(t_n, y_n)
        return t_np1, y_np1

    def BDF2step(self, t_n, y_n, y_nm1, h):
        return self.BDFstep_general(t_n, [y_n, y_nm1], [4, -1, 2], 3, h)
        # t_np1 = t_n + h
        # y_np1 = y_n
        # for i in range(self.maxit):
        #     y_np1_old = y_np1
        #     y_np1 = (4*y_n - 1*y_nm1 + 2*h*self.problem.rhs(t_np1, y_np1))/3

        #     if(np.norm(y_np1-y_np1_old)) < self.tol:
        #         return t_np1, y_np1
        # else:
        #     raise Explicit_ODE_Exception

    def BDF3step(self, t_n, y_n, y_nm1, y_nm2):
        return self.BDFstep_general(t_n, [y_n, y_nm1, y_nm2], [18, -9, 2, 6], 11, h)
        # t_np1 = t_n + h
        # y_np1 = y_n
        # for i in range(self.maxit):
        #     y_np1_old = y_np1
        #     y_np1 = (18*y_n - 9*y_nm1 + 2*y_nm2 + 6*h*self.problem.rhs(t_np1, y_np1))/11
        #     if(np.norm(y_np1-y_np1_old)) < self.tol:
        #         return t_np1, y_np1
        # else:
        #     raise Explicit_ODE_Exception

    def BDFstep_general(self, t_n, Y, multipliers, divisor, h):
        t_np1 = t_n + h
        y_np1 = Y[0]
        for i in range(self.maxit):
            y_np1_old = y_np1
            y_np1 = multipliers[-1]*self.problem.rhs(t_np1, y_np1)
            for i in range(len(Y)-1):
                alpha_i = multipliers[i]
                y_i =  Y[i]
                y_np1 += alpha_i * Y[i]
            y_np1 /= divisor

            if(np.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exceptions


class BDF_4(Explicit_ODE):
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)
        self.maxsteps = 500
        self.maxit = 100
        self.tol = 1.e-8

    def integrate(self, t0, tf, y0):
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
        t, y = self.BDF2step(t, y_list[-1], y_list[-2], y_list[-3], h)
        t_list.append(t)
        y_list.append(y)
        for i in range(3, self.maxsteps):
            if t >= tf:
                break
            # BDF4
            t, y = self.BDF4step(t, y_list[-1], y_list[-2], y_list[-3], y_list[-4], h)
            t_list.append(t)
            y_list.append(y)
        pass

    def EEstep(self, t_n, y_n, h):
        t_np1 = t_n + h
        y_np1 = y_n + h*self.problem.rhs(t_n, y_n)
        return t_np1, y_np1

    def BDF2step(self, t_n, y_n, y_nm1, h):
        return self.BDFstep_general(t_n, [y_n, y_nm1], [4, -1, 2], 3, h)
        # t_np1 = t_n + h
        # y_np1 = y_n
        # for i in range(self.maxit):
        #     y_np1_old = y_np1
        #     y_np1 = (4*y_n - 1*y_nm1 + 2*h*self.problem.rhs(t_np1, y_np1))/3

        #     if(np.norm(y_np1-y_np1_old)) < self.tol:
        #         return t_np1, y_np1
        # else:
        #     raise Explicit_ODE_Exception

    def BDF3step(self, t_n, y_n, y_nm1, y_nm2):
        return self.BDFstep_general(t_n, [y_n, y_nm1, y_nm2], [18, -9, 2, 6], 11, h)
        # t_np1 = t_n + h
        # y_np1 = y_n
        # for i in range(self.maxit):
        #     y_np1_old = y_np1
        #     y_np1 = (18*y_n - 9*y_nm1 + 2*y_nm2 + 6*h*self.problem.rhs(t_np1, y_np1))/11
        #     if(np.norm(y_np1-y_np1_old)) < self.tol:
        #         return t_np1, y_np1
        # else:
        #     raise Explicit_ODE_Exception

    def BDF4step(self, t_n, y_n, y_nm1, y_nm2, y_nm3, h):
        return self.BDFstep_general(t_n, [y_n, y_nm1, y_nm2, y_nm3], [48, -36, 16, -3, 12], 25, h)
        # t_np1 = t_n + h
        # y_np1 = y_n
        # for i in range(self.maxit):
        #     y_np1_old = y_np1
        #     y_np1 = (48*y_n - 36*y_nm1 + 16*y_nm2 - 3*y_nm3 + 12*h*self.problem.rhs(t_np1, y_np1))/25
        #     if(np.norm(y_np1-y_np1_old)) < self.tol:
        #         return t_np1, y_np1
        # else:
        #     raise Explicit_ODE_Exceptions

    def BDFstep_general(self, t_n, Y, multipliers, divisor, h):
        t_np1 = t_n + h
        y_np1 = Y[0]
        for i in range(self.maxit):
            y_np1_old = y_np1
            y_np1 = multipliers[-1]*self.problem.rhs(t_np1, y_np1)
            for i in range(len(Y)-1):
                alpha_i = multipliers[i]
                y_i =  Y[i]
                y_np1 += alpha_i * Y[i]
            y_np1 /= divisor

            if(np.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exceptions