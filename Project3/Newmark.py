from assimulo.ode import Explicit_Problem, Explicit_ODE_Exception, ID_PY_OK
from assimulo.explicit_ode import Explicit_ODE
import numpy as np

import scipy as sp

class Explicit_Problem_2nd(Explicit_Problem):
    def __init__(self, y0, yp0, Mmat, Cmat, Kmat, func):
        self.t0 = 0
        self.y0 = np.hstack((y0, yp0))

        self.n = len(y0)
        self.Mmat = Mmat
        self.Cmat = Cmat
        self.Kmat = Kmat
        self.func = func

    def rhs(self, t, y):
        u = y[:self.n]
        up = y[self.n:]
        upp = sp.sparse.linalg.spsolve(self.Mmat, self.Kmat@u + self.Cmat@up + self.func(t))
        return np.hstack((up, upp))


class Explicit_2nd_Order(Explicit_ODE):
    tol = 1.e-8
    maxit = 100
    maxsteps = 5000

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)

        self.options["h"] = 0.01
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _get_h(self):
        return self.options["h"]

    def _set_h(self, new_h):
        self.options["h"] = float(new_h)

    def integrate(self, t, y, tf, opts):
        u = y[:len(y) // 2]
        up = y[len(y) // 2:]
        upp = self.problem.rhs(t, y)[len(y) // 2:]
        self.statistics["nfcns"] += 1
        y_list = [y]
        t_list = [t]

        h = min([self._get_h(), abs(tf - t)])
        t = t + h
        for i in range(self.maxsteps):
            if t >= tf:
                break
            u, up, upp = self.step(u, up, upp, h, t)
            self.statistics["nsteps"] += 1

            t_list.append(t)
            y_list.append(np.hstack((u, up)))
            h = min([self._get_h(), abs(tf - t)])
            t = t + h

        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, t_list, y_list


class Newmark_Exp(Explicit_2nd_Order):
    def __init__(self, problem, gamma=0.5):
        Explicit_2nd_Order.__init__(self, problem)
        self.gamma = gamma

    def step(self, u, up, upp, h, t):
        u_np1 = u + up * h + 0.5 * upp * h**2
        temp = np.zeros(len(up))
        upp_np1 = self.problem.rhs(t, np.hstack((u, temp)))[len(upp):]
        self.statistics["nfcns"] += 1
        up_np1 = up + upp * h * (1 - self.gamma) + upp_np1 * self.gamma * h
        return u_np1, up_np1, upp_np1


class HHT(Explicit_2nd_Order):
    def __init__(self, problem, alpha=0):
        Explicit_2nd_Order.__init__(self, problem)

        self.alpha = alpha
        self.beta = ((1 - alpha) / 2)**2
        self.gamma = 0.5 - alpha

        self.Mmat = self.problem.Mmat
        self.Cmat = self.problem.Cmat
        self.Kmat = self.problem.Kmat
        self.func = self.problem.func

    def step(self, u, up, upp, h, t):
        gdb = self.gamma / self.beta
        Amat = self.Mmat / (self.beta * h**2) + self.Cmat * (gdb / h) + self.Kmat * (1 + self.alpha)

        term2 = self.Mmat @ (u / (self.beta * h**2) + up / (self.beta * h) + upp * (1 / (2 * self.beta) - 1))
        term3 = self.Cmat @ (u * (gdb / h) - up * (1 - gdb) - upp * h * (1 - gdb / 2))
        term4 = self.alpha * self.Kmat @ u

        u_np1 = sp.sparse.linalg.spsolve(Amat, self.problem.func(t) + term2 + term3 + term4)
        self.statistics["nfcns"] += 1
        up_np1 = ((u_np1 - u) / h) * gdb + up * (1 - gdb) + upp * h * (1 - gdb / 2)
        upp_np1 = (u_np1 - u) / (self.beta * h**2) - up / (self.beta * h) - upp * (1 / (2 * self.beta) - 1 )
        return u_np1, up_np1, upp_np1


class Newmark(HHT):
    def __init__(self, problem, beta=0.5, gamma=0.5):
        HHT.__init__(self, problem, alpha=0)
        self.beta = beta
        self.gamma = gamma

        if not (0.5 <= gamma <= 2 * beta):
            print("Method is not unconditionally stable.")

        if (gamma == 2 * beta == 0.5):
            print("Method is second order accurate.")
        else:
            print("Method is first order accurate.")


if __name__ == "__main__":
    # When run as main, compares the methods to CVode using the function 'problem_func' as example.
    def problem_func(t, y):
        temp = spring_constant * (1 - 1 / np.sqrt(y[0]**2 + y[1]**2))
        return np.asarray([y[2], y[3], -y[0] * temp, -y[1] * temp - 1])
    spring_constant = 3

    starting_point = np.array([1 - 1e-6, 0, 0, 0])
    problem = Explicit_Problem(problem_func, y0=starting_point)
    t_end = 3

    from assimulo.solvers.sundials import CVode
    problem.name = "CVode"
    solver = CVode(problem)
    solver.simulate(t_end)
    solver.plot()

    problem.name = "Explicit Newmark solver"
    solver = Newmark_Exp(problem)
    t, y = solver.simulate(t_end)
    solver.plot()

    problem.name = "Implicit Newmark solver"
    solver = Newmark(problem)
    t, y = solver.simulate(t_end)
    solver.plot()
