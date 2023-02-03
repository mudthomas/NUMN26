from assimulo.solvers.sundials import CVode
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import Explicit_Problem, Explicit_ODE_Exception, ID_PY_OK
import numpy as np
import scipy.optimize as opt


class BDF(Explicit_ODE):
    maxsteps = 500
    maxit = 100
    tol = 1.e-8

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)
        # Solver options
        self.options["h"] = 0.01
        self.rtol = 1.0e-6
        self.atol = 1.0e-6*np.ones(len(problem.y0))

        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _set_h(self, h):
        self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    def EEstep(self, t_n, y_n, h):
        self.statistics["nfcns"] += 1
        return t_n + h, y_n + h*self.problem.rhs(t_n, y_n)

    def integrate(self, t0, y0, tf, opts):
        pass


class BDF2(BDF):
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
            t, y = self.BDF2step(t, y_list[-2:][::-1], h)
            t_list.append(t)
            y_list.append(y)
            h = min(self._get_h(), np.abs(tf-t))

        return ID_PY_OK, t_list, y_list

    def BDF2step(self, t_n, Y, h):
        coeffs = [4/3, -1/3, 2/3]
        static_terms = coeffs[0] * Y[0]
        for i in range(1, len(Y)):
            static_terms += coeffs[i] * Y[i]

        t_np1 = t_n + h
        y_np1_i = Y[0]
        for i in range(self.maxit):
            y_np1_ip1 = static_terms + coeffs[-1]*h*self.problem.rhs(t_np1, y_np1_i)
            self.statistics["nfcns"] += 1
            if(np.linalg.norm(y_np1_ip1-y_np1_i)) < self.tol:
                return t_np1, y_np1_ip1
            y_np1_i = y_np1_ip1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


class BDF3(BDF2):
    def integrate(self, t0, y0, tf, opts):
        h = min(self._get_h(), np.abs(tf-t0))
        t_list = [t0]
        y_list = [y0]

        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        h = min(self._get_h(), np.abs(tf-t0))

        self.statistics["nsteps"] += 1
        t, y = self.BDF2step(t, y_list[-2:][::-1], h)
        t_list.append(t)
        y_list.append(y)
        h = min(self._get_h(), np.abs(tf-t0))

        for i in range(2, self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            t, y = self.BDF3step(t, y_list[-3:][::-1], h)
            t_list.append(t)
            y_list.append(y)
            h = min(self._get_h(), np.abs(tf-t))

        return ID_PY_OK, t_list, y_list

    def BDF3step(self, t_n, Y, h):
        coeffs = [18/11, -9/11, 2/11, 6/11]
        static_terms = coeffs[0] * Y[0]
        for i in range(1, len(Y)):
            static_terms += coeffs[i] * Y[i]

        t_np1 = t_n + h
        y_np1_i = Y[0]
        for i in range(self.maxit):
            y_np1_ip1 = static_terms + coeffs[-1]*h*self.problem.rhs(t_np1, y_np1_i)
            self.statistics["nfcns"] += 1
            if(np.linalg.norm(y_np1_ip1-y_np1_i)) < self.tol:
                return t_np1, y_np1_ip1
            y_np1_i = y_np1_ip1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


class BDF4(BDF3):
    def integrate(self, t0, y0, tf, opts):
        h = min(self._get_h(), np.abs(tf-t0))
        t_list = [t0]
        y_list = [y0]

        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        h = min(self._get_h(), np.abs(tf-t0))

        self.statistics["nsteps"] += 1
        t, y = self.BDF2step(t, y_list[-2:][::-1], h)
        t_list.append(t)
        y_list.append(y)
        h = min(self._get_h(), np.abs(tf-t0))

        for i in range(3, self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            t, y = self.BDF4step(t, y_list[-4:][::-1], h)
            t_list.append(t)
            y_list.append(y)
            h = min(self._get_h(), np.abs(tf-t))

        return ID_PY_OK, t_list, y_list

    def BDF4step(self, t_n, Y, h):
        coeffs = [48/25, -36/25, 16/25, -3/25, 12/25]
        static_terms = coeffs[0] * Y[0]
        for i in range(1, len(Y)):
            static_terms += coeffs[i] * Y[i]

        t_np1 = t_n + h
        y_np1_i = Y[0]
        for i in range(self.maxit):
            y_np1_ip1 = static_terms + coeffs[-1]*h*self.problem.rhs(t_np1, y_np1_i)
            self.statistics["nfcns"] += 1
            if(np.linalg.norm(y_np1_ip1-y_np1_i)) < self.tol:
                return t_np1, y_np1_ip1
            y_np1_i = y_np1_ip1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


class BDF2_Newton(BDF2):
    maxit = 10000
    rtol = 1.0e-6
    atol = np.array([1.0e-4, 1.0e-4, 1.0e-2, 1.0e-2])

    def integrate(self, t0, y0, tf, opts):
        h_list = []
        t_list = [t0]
        y_list = [y0]

        h = min(self._get_h(), np.abs(tf-t0))
        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        h_list.append(h)

        h = min(h, np.abs(tf-t))
        while(self.statistics["nsteps"] < self.maxsteps-1):
            if h < self.tol:
                print("h too small.")
                break
            if t >= tf:
                break
            t, y, h_new = self.BDF2step_Newton(t, y_list[-2:][::-1], h, h_list[-1:])

            t_list.append(t)
            y_list.append(y)
            h_list.append(h_new)
            self.statistics["nsteps"] += 1
            h = min(h_new, np.abs(tf-t))

        return ID_PY_OK, t_list, y_list

    def getJacobian(self, t_np1, point, evaluation, finite_diff, Newton_func):
        jacobian = np.zeros((len(point), len(point)))
        for j in range(len(point)):
            shift = np.zeros(len(point))
            shift[j] = finite_diff
            jacobian[:, j] = (Newton_func(t_np1, point + shift) - evaluation)/finite_diff
            self.statistics["nfcns"] += 1
        return jacobian

    def getNorm(self, y):
        ret = 0
        for i in range(len(y)):
            W = self.rtol * np.abs(y[i]) + self.atol[i]
            ret += (y[i]/W)**2
        ret /= len(y)
        return np.sqrt(ret)

    def BDF2step_Newton(self, t_n, Y, h0, H):
        h_nm1 = H[0]
        y_n, y_nm1 = Y

        def NewtonFunc(t, y):
            a_np1 = (h_nm1+2*h)/(h*(h_nm1 + h))
            a_nm1 = h/(h_nm1 * (h + h_nm1))
            a_n = - a_np1 - a_nm1
            return a_np1 * y + a_n*y_n + a_nm1*y_nm1 - self.problem.rhs(t, y)

        h = h0
        t_np1 = t_n + h
        y_np1_i = Y[0]

        temp = NewtonFunc(t_np1, y_np1_i)
        finite_diff = 1e-8
        jacobian = self.getJacobian(t_np1, y_np1_i, temp, finite_diff, NewtonFunc)
        new_jacobian = 1

        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            y_np1_ip1 = y_np1_i - np.linalg.solve(jacobian, y_np1_i)
            convergence = self.getNorm(y_np1_ip1)/self.getNorm(y_np1_i) < 1

            if convergence:
                new_jacobian = 0
                if(np.linalg.norm(y_np1_ip1-y_np1_i)) <= self.tol:
                    return t_np1, y_np1_ip1, h
                else:
                    y_np1_i = y_np1_ip1
            else:
                if new_jacobian == 1:
                    h /= 2
                    t_np1 = t_n + h
                    temp = NewtonFunc(t_np1, y_np1_i)
                    jacobian = self.getJacobian(t_np1, y_np1_i, temp, finite_diff, NewtonFunc)
                else:
                    temp = NewtonFunc(t_np1, y_np1_i)
                    jacobian = self.getJacobian(t_np1, y_np1_i, temp, finite_diff, NewtonFunc)
                    new_jacobian = 1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


class BDF3_Newton(BDF2_Newton):
    maxit = 10000
    rtol = 1.0e-6
    atol = np.array([1.0e-4, 1.0e-4, 1.0e-2, 1.0e-2])

    def integrate(self, t0, y0, tf, opts):
        h_list = []
        t_list = [t0]
        y_list = [y0]

        h = min(self._get_h(), np.abs(tf-t0))
        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        h_list.append(h)

        h = min(h, np.abs(tf-t0))
        self.statistics["nsteps"] += 1
        t, y = self.BDF2step(t, y_list[-2:][::-1], h)
        t_list.append(t)
        y_list.append(y)
        h_list.append(h)

        h = min(h, np.abs(tf-t))

        while(self.statistics["nsteps"] < self.maxsteps-2):
            if h < self.tol:
                print("h too small.")
                break
            if t >= tf:
                break
            t, y, h_new = self.BDF3step_Newton(t, y_list[-3:][::-1], h, h_list[-2:][::-1])

            # _, y2, _ = self.BDF2step_Newton(t, y_list[-2:][::-1], h, h_list[-1:][::-1])
            # error = self.getNorm(y-y2)
            # if error < self.tol:
            #     t_list.append(t)
            #     y_list.append(y)
            #     h_list.append(h_new)
            #     self.statistics["nsteps"] += 1
            #     h = min(h_new*2, np.abs(tf-t))
            # else:
            #     h = min(h_new/2, np.abs(tf-t))
            t_list.append(t)
            y_list.append(y)
            h_list.append(h_new)
            self.statistics["nsteps"] += 1
            h = min(h_new, np.abs(tf-t))

        return ID_PY_OK, t_list, y_list

    def getJacobian(self, t_np1, point, evaluation, finite_diff, Newton_func):
        jacobian = np.zeros((len(point), len(point)))
        for j in range(len(point)):
            shift = np.zeros(len(point))
            shift[j] = finite_diff
            jacobian[:, j] = (Newton_func(t_np1, point + shift) - evaluation)/finite_diff
            self.statistics["nfcns"] += 1
        return jacobian

    def BDF3step_Newton(self, t_n, Y, h0, H):
        h_nm1, h_nm2 = H
        y_n, y_nm1, y_nm2 = Y

        def NewtonFunc(t, y):
            a_n = - ((h + h_nm1) * (h + h_nm1 + h_nm2))/(h*h_nm1*(h_nm1+h_nm2))
            a_nm1 = (h * (h + h_nm1 + h_nm2))/(h_nm1 * h_nm2 * (h + h_nm1))
            a_nm2 = - (h * (h + h_nm1))/(h_nm2 * (h_nm1 + h_nm2) * (h + h_nm1 + h_nm2))
            a_np1 = - (a_n + a_nm1 + a_nm2)
            return a_np1*y + a_n*y_n + a_nm1*y_nm1 + a_nm2*y_nm2 - self.problem.rhs(t, y)

        h = h0
        t_np1 = t_n + h
        y_np1_i = Y[0]

        temp = NewtonFunc(t_np1, y_np1_i)
        finite_diff = 1e-8
        jacobian = self.getJacobian(t_np1, y_np1_i, temp, finite_diff, NewtonFunc)
        new_jacobian = 1

        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            y_np1_ip1 = y_np1_i - np.linalg.solve(jacobian, y_np1_i)
            convergence = self.getNorm(y_np1_ip1)/self.getNorm(y_np1_i) <= 1

            if convergence:
                new_jacobian = 0
                if(np.linalg.norm(y_np1_ip1-y_np1_i)) <= self.tol:
                    return t_np1, y_np1_ip1, h
                else:
                    y_np1_i = y_np1_ip1
            else:
                if new_jacobian == 1:
                    h /= 2
                    t_np1 = t_n + h
                    temp = NewtonFunc(t_np1, y_np1_i)
                    jacobian = self.getJacobian(t_np1, y_np1_i, temp, finite_diff, NewtonFunc)
                else:
                    temp = NewtonFunc(t_np1, y_np1_i)
                    jacobian = self.getJacobian(t_np1, y_np1_i, temp, finite_diff, NewtonFunc)
                    new_jacobian = 1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


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

        for spring_constant in [5]:
            starting_point = np.array([1, 0, 0, 0])
            problem = Explicit_Problem(problem_func, y0=starting_point)
            t_end = 10

            problem.name = f"Task 3, k={spring_constant}, CVode"
            solver = CVode(problem)
            solver.simulate(t_end)
            solver.plot()

            # problem.name = f"Task 3, k={spring_constant}, EE-solver"
            # solver = EE_solver(problem)
            # solver.simulate(t_end)
            # solver.plot()

            # problem.name = f"Task 3, k={spring_constant}, BDF2-solver, FPI"
            # solver = BDF2(problem)
            # solver.simulate(t_end)
            # solver.plot()

            # problem.name = f"Task 3, k={spring_constant}, BDF3-solver, FPI"
            # solver = BDF3(problem)
            # solver.simulate(t_end)
            # solver.plot()

            # problem.name = f"Task 3, k={spring_constant}, BDF4-solver, FPI"
            # solver = BDF4(problem)
            # solver.simulate(t_end)
            # solver.plot()

            problem.name = f"Task 3, k={spring_constant}, BDF2-solver, Newton"
            solver = BDF2_Newton(problem)
            solver.simulate(t_end)
            solver.plot()

            problem.name = f"Task 3, k={spring_constant}, BDF3-solver, Newton"
            solver = BDF3_Newton(problem)
            solver.simulate(t_end)
            solver.plot()

    # doTask1()
    doTask3()
