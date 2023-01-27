from assimulo.solvers.sundials import CVode
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import Explicit_Problem, Explicit_ODE_Exception, ID_PY_OK
import numpy as np


class BDF_general(Explicit_ODE):
    maxsteps = 500
    maxit = 1000
    tol = 1.e-8

    def __init__(self, problem, degree):
        Explicit_ODE.__init__(self, problem)
        self.degree = degree

        # Solver options
        self.options["h"] = 0.01
        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _set_h(self, h):
        self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    def integrate(self, t0, y0, tf, opts):
        h = self.options["h"]
        h = min(h, np.abs(tf-t0))

        t_list = [t0]
        y_list = [y0]

        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)

        for i in range(1, self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            t, y = self.BDFstep_general(t, y_list[-self.degree:][::-1], h)

            t_list.append(t)
            y_list.append(y)
            h = min(h, np.abs(tf-t))

        return ID_PY_OK, t_list, y_list

    def EEstep(self, t_n, y_n, h):
        self.statistics["nfcns"] += 1
        return t_n + h, y_n + h*self.problem.rhs(t_n, y_n)

    def BDFstep_general(self, t_n, Y, h):
        coeffs = [[1, -1, -1],
                  [3, -4, 1, -2],
                  [11, -18, 9, -2, -6],
                  [25, -48, 36, -16, 3, -12],
                  [137, -300, 300, -200, 75, -12, -60],
                  [147, -360, 450, -400, 225, -72, 10, -60]]
        return self.BDFstep_general_internal(t_n, Y, coeffs[len(Y)-1], h)

    def BDFstep_general_internal(self, t_n, Y, coeffs, h):
        t_np1 = t_n + h
        y_np1 = Y[0]
        static_terms = np.zeros(len(Y[0]))

        for i in range(len(Y)):
            static_terms += coeffs[i+1] * Y[i]

        # FPI
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            y_np1_old = y_np1
            y_np1 = - static_terms/coeffs[0] - (coeffs[-1]/coeffs[0])*h*self.problem.rhs(t_np1, y_np1_old)
            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


class BDF_general_newton(BDF_general):
    maxsteps = 500
    maxit = 1000
    tol = 1.e-8

    def BDFstep_general_internal(self, t_n, Y, coeffs, h):
        t_np1 = t_n + h
        y_np1 = Y[0]
        static_terms = np.zeros(len(Y[0]))

        for i in range(len(Y)):
            static_terms += coeffs[i+1] * Y[i]

        # Newton
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            y_np1_old = y_np1
            # Newton predictor
            feval = self.problem.rhs(t_n, y_np1_old)
            y_np1 = y_np1_old - feval/((self.problem.rhs(t_n, y_np1_old+1e-6)-feval)/1e-6)
            # BDF corrector
            y_np1 = - (static_terms + coeffs[-1]*h*feval)/coeffs[0]
            if(np.linalg.norm(y_np1-y_np1_old)) < self.tol:
                return t_np1, y_np1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


class BDF_2(BDF_general):
    def __init__(self, problem):
        BDF_general.__init__(self, problem, 2)


class BDF_3(BDF_general):
    def __init__(self, problem):
        BDF_general.__init__(self, problem, 3)


class BDF_4(BDF_general):
    def __init__(self, problem):
        BDF_general.__init__(self, problem, 4)


class BDF_3_newton(BDF_general_newton):
    def __init__(self, problem):
        BDF_general.__init__(self, problem, 3)


class BDF_4_newton(BDF_general_newton):
    def __init__(self, problem):
        BDF_general.__init__(self, problem, 4)


class BDF_5(BDF_general):
    def __init__(self, problem):
        BDF_general.__init__(self, problem, 5)


class BDF_6(BDF_general):
    def __init__(self, problem):
        BDF_general.__init__(self, problem, 6)


class EE_solver(Explicit_ODE):
    maxsteps = 500
    maxit = 100
    tol = 1.e-8

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)

        # Solver options
        self.options["h"] = 0.01
        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _set_h(self, h):
        self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    def integrate(self, t, y, tf, opts):
        h = min(self.options["h"], np.abs(tf-t))
        t_list = [t]
        y_list = [y]
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            t, y = self.EEstep(t, y, h)
            t_list.append(t)
            y_list.append(y)
            h = min(h, np.abs(tf-t))
        return ID_PY_OK, t_list, y_list

    def EEstep(self, t_n, y_n, h):
        self.statistics["nfcns"] += 1
        return t_n + h, y_n + h*self.problem.rhs(t_n, y_n)


if __name__ == "__main__":
    def doTask1():
        def problem_func(t, y):
            temp = k * (1 - 1/np.sqrt(y[0]**2+y[1]**2))
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
            temp = k * (1 - 1/np.sqrt(y[0]**2+y[1]**2))
            return np.asarray([y[2], y[3], -y[0]*temp, -y[1]*temp - 1])

        for k in [10, 100, 1000]:
            starting_point = np.array([2, 0, 0, 0])
            problem = Explicit_Problem(problem_func, y0=starting_point)

            # problem.name = f"Task 3, k={k}, CVode"
            # solver = CVode(problem)
            # solver.simulate(100)
            # solver.plot()

            # problem.name = f"Task 3, k={k}, EE-solver"
            # solver = EE_solver(problem)
            # solver.simulate(100)
            # solver.plot()

            # problem.name = f"Task 3, k={k}, BDF2-solver"
            # solver = BDF_2(problem)
            # solver.simulate(1000)
            # solver.plot()

            # problem.name = f"Task 3, k={k}, BDF3-solver"
            # solver = BDF_3_newton(problem)
            # solver.simulate(100)
            # solver.plot()

            problem.name = f"Task 3, k={k}, BDF4-solver"
            solver = BDF_4_newton(problem)
            solver.simulate(100)
            solver.plot()
            solver.print_statistics()

            # problem.name = f"Task 3, k={k}, BDF6-solver"
            # solver = BDF_6(problem)
            # solver.simulate(100)
            # solver.plot()

    # doTask1()
    doTask3()
