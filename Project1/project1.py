from assimulo.solvers.sundials import CVode
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import Explicit_Problem, Explicit_ODE_Exception, ID_PY_OK
import numpy as np
import scipy.optimize as opt


class BDF_general(Explicit_ODE):
    maxsteps = 500
    maxit = 100
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
        h = min(self._get_h(), np.abs(tf-t0))
        t_list = [t0]
        y_list = [y0]

        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        h = min(self._get_h(), np.abs(tf-t0))

        for i in range(1, self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            t, y = self.BDFstep_general(t, y_list[-self.degree:][::-1], h)
            t_list.append(t)
            y_list.append(y)
            h = min(self._get_h(), np.abs(tf-t))

        return ID_PY_OK, t_list, y_list

    def EEstep(self, t_n, y_n, h):
        self.statistics["nfcns"] += 1
        return t_n + h, y_n + h*self.problem.rhs(t_n, y_n)

    def BDFstep_general(self, t_n, Y, h):
        coeffs = [[1, 1],
                  [4/3, -1/3, 2/3],
                  [18/11, -9/11, 2/11, 6/11],
                  [48/25, -36/25, 16/25, -3/25, 12/25],
                  [300/137, -300/137, 200/137, -75/137, 12/137, 60/137],
                  [360/147, -450/147, 400/147, -225/147, 72/147, -10/147, 60/147]]
        return self.BDFstep_general_internal(t_n, Y, coeffs[len(Y)-1], h)

    def BDFstep_general_internal(self, t_n, Y, coeffs, h):
        static_terms = coeffs[0] * Y[0]
        for i in range(1, len(Y)):
            static_terms += coeffs[i] * Y[i]

        # Predictor
        t_np1 = t_n + h
        y_np1_i = Y[0]
        for i in range(self.maxit):
            # BDF Evaluator
            y_np1_temp = static_terms + coeffs[-1]*h*self.problem.rhs(t_np1, y_np1_i)
            self.statistics["nfcns"] += 1
            # FPI Corrector
            y_np1_ip1 = y_np1_temp

            if(np.linalg.norm(y_np1_ip1-y_np1_i)) < self.tol:
                return t_np1, y_np1_ip1
            y_np1_i = y_np1_ip1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


class BDF_general_newton(BDF_general):
    maxsteps = 500
    maxit = 100
    tol = 1.e-8

    def __init__(self, problem, degree):
        Explicit_ODE.__init__(self, problem)
        self.degree = degree
        # Solver options
        self.options["h"] = 0.001
        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def integrate(self, t0, y0, tf, opts):
        h = min(self._get_h(), np.abs(tf-t0))
        t_list = [t0]
        y_list = [y0]

        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        h = min(self._get_h(), np.abs(tf-t0))


        for i in range(1, self.degree):
            if t >= tf:
                break
            t, y = self.BDFstep_general_FPI(t, y_list[::-1], h)
            t_list.append(t)
            y_list.append(y)
            self.statistics["nsteps"] += 1
            h = min(self._get_h(), np.abs(tf-t))

        while(self.statistics["nsteps"] < self.maxsteps-1):
            if h < self.tol:
                return ID_PY_OK, t_list, y_list
            if t >= tf:
                return ID_PY_OK, t_list, y_list
            try:
                t, y = self.BDFstep_general(t, y_list[-self.degree:][::-1], h)
                t_list.append(t)
                y_list.append(y)
                self.statistics["nsteps"] += 1
                print("success!")
            except RuntimeError:
                print(self.statistics["nsteps"])
                print(f"h before = {h}")
                self._set_h(h/2)
                h = min(self._get_h(), np.abs(tf-t))
                print(f"h after = {h}")

        return ID_PY_OK, t_list, y_list

    def BDFstep_general_FPI(self, t_n, Y, h):
        coeffs = [[1, 1],
                  [4/3, -1/3, 2/3],
                  [18/11, -9/11, 2/11, 6/11],
                  [48/25, -36/25, 16/25, -3/25, 12/25],
                  [300/137, -300/137, 200/137, -75/137, 12/137, 60/137],
                  [360/147, -450/147, 400/147, -225/147, 72/147, -10/147, 60/147]]
        return self.BDFstep_general_internal_FPI(t_n, Y, coeffs[len(Y)-1], h)

    def BDFstep_general_internal_FPI(self, t_n, Y, coeffs, h):
        static_terms = coeffs[0] * Y[0]
        for i in range(1, len(Y)):
            static_terms += coeffs[i] * Y[i]

        # Predictor
        t_np1 = t_n + h
        y_np1_i = Y[0]
        for i in range(self.maxit):
            # BDF Evaluator
            y_np1_temp = static_terms + coeffs[-1]*h*self.problem.rhs(t_np1, y_np1_i)
            self.statistics["nfcns"] += 1
            # FPI Corrector
            y_np1_ip1 = y_np1_temp

            if(np.linalg.norm(y_np1_ip1-y_np1_i)) < self.tol:
                return t_np1, y_np1_ip1
            y_np1_i = y_np1_ip1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")

    def getJacobian(self, point, evaluation, finite_diff, Newton_func):
        jacobian = np.zeros((len(point), len(point)))
        for j in range(len(point)):
            shift = np.zeros(len(point))
            shift[j] = finite_diff
            jacobian[:, j] = (Newton_func(point + shift) - evaluation)/finite_diff
            self.statistics["nfcns"] += 1
        return jacobian

    def BDFstep_general_internal(self, t_n, Y, coeffs, h):
        static_terms = coeffs[0] * Y[0]
        for i in range(1, len(Y)):
            static_terms += coeffs[i] * Y[i]

        Newton_func = lambda x: (static_terms + coeffs[-1]*h*self.problem.rhs(t_np1, x)) - x

        # Predict
        t_np1 = t_n + h
        y_np1_i = Y[0]


        finite_diff = 1e-8
        temp = Newton_func(Y[0])
        self.statistics["nfcns"] += 1

        jacobian = self.getJacobian(Y[0], temp, finite_diff, Newton_func)
        new_jacobian = 1

        for i in range(self.maxit):
            if h<1e-14:
                print("h too small")
                break
            # BDF evaluator
            # temp = (static_terms + coeffs[-1]*h*self.problem.rhs(t_np1, y_np1_i))
            self.statistics["nfcns"] += 1

            y_np1_ip1 = y_np1_i - np.linalg.solve(jacobian, y_np1_i)
            convergence = np.linalg.norm(y_np1_i - np.linalg.solve(jacobian, y_np1_i))/np.linalg.norm(y_np1_i) < 1
            print(np.linalg.norm(y_np1_ip1-y_np1_i))
            if convergence:
                print("####\nconvergence")
                new_jacobian = 0
                if(np.linalg.norm(y_np1_ip1-y_np1_i)) <= self.tol:
                    return t_np1, y_np1_ip1
                else:
                    y_np1_i = y_np1_ip1
            else:
                print("####\nnot convergence")
                if new_jacobian == 0:
                    temp = Newton_func(y_np1_i)
                    jacobian = self.getJacobian(y_np1_i, temp, finite_diff, Newton_func)
                    new_jacobian = 1
                else:
                    h /= 2
                    temp = Newton_func(y_np1_i)
                    jacobian = self.getJacobian(y_np1_i, temp, finite_diff, Newton_func)
                    new_jacobian = 1


        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {self.maxit} iterations")

        # fsolve
        # Newton_func = lambda x: (static_terms + coeffs[-1]*h*self.problem.rhs(t_np1, x)) - x
        # y_np1_ip1, info = opt.fsolve(Newton_func, y_np1_i, xtol=self.tol, full_output=True)[:2]
        # self.statistics["nfcns"] += info["nfev"]
        # return t_np1, y_np1_ip1




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


class EE_solver(Explicit_ODE):
    maxsteps = 500
    maxit = 10
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

        for spring_constant in [1]:
            starting_point = np.array([1-1e-6, 0, 0, 0])
            problem = Explicit_Problem(problem_func, y0=starting_point)
            t_end = 1

            # problem.name = f"Task 3, k={spring_constant}, CVode"
            # solver = CVode(problem)
            # solver.simulate(t_end)
            # solver.plot()

            # problem.name = f"Task 3, k={spring_constant}, EE-solver"
            # solver = EE_solver(problem)
            # solver.simulate(t_end)
            # solver.plot()

            # problem.name = f"Task 3, k={spring_constant}, BDF2-solver"
            # solver = BDF_2(problem)
            # solver.simulate(t_end)
            # solver.plot()

            problem.name = f"Task 3, k={spring_constant}, BDF3-solver, FPI"
            solver = BDF_3(problem)
            solver.simulate(t_end)
            solver.plot()

            # problem.name = f"Task 3, k={spring_constant}, BDF4-solver, FPI"
            # solver = BDF_4(problem)
            # solver.simulate(t_end)
            # solver.plot()

            problem.name = f"Task 3, k={spring_constant}, BDF3-solver, Newton"
            solver = BDF_3_newton(problem)
            solver.simulate(t_end)
            solver.plot()

            # problem.name = f"Task 3, k={spring_constant}, BDF4-solver, Newton"
            # solver = BDF_4_newton(problem)
            # solver.simulate(t_end)
            # solver.plot()
            # solver.print_statistics()

    # doTask1()
    doTask3()
