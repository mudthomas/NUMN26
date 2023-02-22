from assimulo.ode import Explicit_Problem, Explicit_ODE_Exception, ID_PY_OK
from assimulo.explicit_ode import Explicit_ODE
import numpy as np


class Explicit_Problem_2nd(Explicit_Problem):
    def __init__(self, y0, yp0, Mmat, Cmat, Kmat, func):
        self.t0 = 0
        self.y0 = np.hstack((y0, yp0))

        self.n = len(y0)
        self.Mmat = Mmat
        self.Cmat = Cmat
        self.KMat = Kmat
        self.func = func

    def rhs(self, t, y):
        u = y[:self.n]
        up = y[self.n:]
        upp = np.solve(self.Mmat, np.dot(self.Kmat, u) + np.dot(self.Cmat, up) + self.func(t))
        return np.hstack((up, upp))


class Explicit_2nd_Order(Explicit_ODE):
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)

        self.options["h"] = 0.01
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _get_h(self):
        return self.options["h"]

    def _set_h(self, new_h):
        self.options["h"] = float(new_h)


class Newmark_Exp(Explicit_2nd_Order):
    tol = 1.e-8
    maxit = 100
    maxsteps = 5000

    def __init__(self, problem, gamma=0.5):
        Explicit_2nd_Order.__init__(self, problem)
        self.gamma = gamma

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
            self.statistics["nsteps"] += 1
            u_np1 = u + up * h + 0.5 * upp * h**2
            temp = np.zeros(len(up))
            upp_np1 = self.problem.rhs(t, np.hstack((u, temp)))[len(upp):]
            self.statistics["nfcns"] += 1
            up_np1 = up + upp * h * (1 - self.gamma) + upp_np1 * self.gamma * h
            u, up, upp = u_np1, up_np1, upp_np1
            t_list.append(t)
            y_list.append(np.hstack((u, up)))
            h = min([self._get_h(), abs(tf - t)])
            t = t + h

        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, t_list, y_list


class Newmark(Explicit_2nd_Order):
    tol = 1.e-8
    maxit = 100
    maxsteps = 5000

    def __init__(self, problem, beta=0.5, gamma=0.5):
        Explicit_2nd_Order.__init__(self, problem)
        self.beta = beta
        self.gamma = gamma

        self.Mmat = self.problem.Mmat
        self.Cmat = self.problem.Cmat
        self.KMat = self.problem.Kmat
        self.func = self.problem.func

    def integrate(self, t, y, tf, opts):
        gdb = self.gamma / self.beta
        u = y[:len(y) // 2]
        up = y[len(y) // 2:]
        upp = self.problem.rhs(t, y)[len(y) // 2:]
        self.statistics["nfcns"] += 1

        y_list = [y]
        t_list = [t]

        h = min([self._get_h(), abs(tf - t)])
        t = t + h
        A = self.Kmat + (self.Mmat / h + self.gamma * self.Cmat) / (self.beta * h)
        ## A should be redone for last step.

        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            temp = upp * (1 / (2 * self.beta) - 1) + (u / h + up) / (self.beta * h)
            term2 = np.dot(self.Mmat, temp)

            temp = (gdb / h) * u - (1 - gdb) * up - h * (1 - gdb / 2) * upp
            term3 = np.dot(self.Cmat, temp)

            u_np1 = np.linalg.solve(A, self.problem.func(t) + term2 + term3)
            self.statistics["nfcns"] += 1

            up_np1 = ((u_np1 - u) / h) * gdb + up * (1 - gdb) + upp * h * (1 - gdb / 2)

            upp_np1 = (((u_np1 - u) / h) - up) / (h * self.beta) - (1 / (2 * self.beta)) * upp

            u, up, upp = u_np1, up_np1, upp_np1
            y_list.append()
            t_list.append(t)
            h = min([self._get_h(), abs(tf - t)])
            t = t + h

        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, t_list, y_list


class HHT(Explicit_2nd_Order):
    pass


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
