from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import Explicit_Problem, Explicit_ODE_Exception, ID_PY_OK
import numpy as np

import BDF_FPI as FPI


class BDF_general(FPI.BDF_general):
    """A solver for ODEs using the BDF method with Newton as corrector.
       Implements a BDF solver with FPI.
    """
    maxsteps = 500
    maxit = 10000
    tol = 1.e-8

    def __init__(self, problem, order):
        """Initiates an instance of BDF_general.

        Args:
            problem (Explicit_Problem): See Assimulo documentation.
            order (int): The order of BDF to use.
        """
        Explicit_ODE.__init__(self, problem)
        self.order = order
        self.options["h"] = 0.01
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
        self.rtol = self.tol
        self.atol = np.ones(len(problem.y0))*self.tol

    def _set_h(self, h):
        """Set the step size.

        Args:
            h (float-like): New step size.
        """
        self.options["h"] = float(h)

    def _get_h(self):
        """Returns the step size.

        Returns:
            float: The current step size.
        """
        return self.options["h"]

    def getJacobian(self, t_np1, point, Newton_func, evaluation=None, finite_diff=1.e-8):
        """Finds the jacobian using finite differences.

        Args:
            t_np1 (float): The evaluation time for the Jacobian.
            point ([floats]): The point at which to find the Jacobian.
            Newton_func (callable): A function of two variables.
            evaluation (arraylike, optional): An evaluation of Newton_fun at point. Defaults to None.
            finite_diff (float, optional): The proverbial difference. Defaults to 1.e-8.

        Returns:
            ndarray: matrix of floats
        """
        if evaluation is None:
            evaluation = Newton_func(t_np1, point)
            self.statistics["nfcns"] += 1

        jacobian = np.zeros((len(point), len(point)))
        for j in range(len(point)):
            shift = np.zeros(len(point))
            shift[j] = finite_diff
            jacobian[:, j] = (Newton_func(t_np1, point + shift) - evaluation)/finite_diff
            self.statistics["nfcns"] += 1
        return jacobian

    def getNorm(self, y):
        """Returns the norm of y.

        Args:
            y (ndarray([floats])): Vector to find the norm of.

        Returns:
            float: The norm of y.
        """
        ret = 0
        for i in range(len(y)):
            W = self.rtol * np.abs(y[i]) + self.atol[i]
            ret += (y[i]/W)**2
        ret /= len(y)
        return np.sqrt(ret)

    def integrate(self, t0, y0, tf, opts):
        """Solves the ODE problem.
        First it takes one step with the explicit Euler method,
        then it takes steps using the BDF method with FPI of increasing order util self.order is reached.
        After self.order number of steps it continues taking steps of that BDF method with Newton util reaching tf.

        Args:
            t0 (float-like): The starting time
            y0 (array([floats])): The starting point
            tf (_type_): The end time
            opts (_type_): _description_

        Returns:
            (ID_PY_OK, [floats], [floats]): ID_PY_OK, a list of the evaluation times and a list of the values.
        """
        h_list = []  # Is this neccessary? Better to calculate from the t's when needed?
        t_list = [t0]
        y_list = [y0]

        h = min(self._get_h(), np.abs(tf-t0))
        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        h_list.append(h)

        h = min(h, np.abs(tf-t))
        for i in range(1, self.order):
            self.statistics["nsteps"] += 1
            t, y = self.BDFstep_general(t, y_list[::-1], h)
            t_list.append(t)
            y_list.append(y)
            h_list.append(h)
            h = min(h, np.abs(tf-t))

        while(self.statistics["nsteps"] < self.maxsteps-1):
            if h < 1.e-14:
                print("h too small.")
                break
            if t >= tf:
                break
            t_new, y_new, h = self.BDFstep_Newton(t, y_list[-self.order:][::-1], h, h_list[-(self.order-1):][::-1])

            if t_new is not None:
                t = t_new
                y = y_new
                t_list.append(t)
                y_list.append(y)
                h_list.append(h)
                self.statistics["nsteps"] += 1
                h = min(h, np.abs(tf-t))

        return ID_PY_OK, t_list, y_list

    def BDFstep_Newton(self, t_n, Y, h, H):
        """One step of the BDF-method using Newtons method as a corrector.

        Args:
            t_n (float): Last evaluation time.
            Y ([floats]): Previous values.
            h (float): Step size.
            H ([floats]): Previous step sizes (one less than Y).

        Raises:
            Explicit_ODE_Exception: See Assimulo documentation.

        Returns:
            (float, float, float): The next evaluation time, its value and the stepsize used.
        """
        NewtonFunc_hvar = self.getNewtonFunc(Y, H)
        t_np1 = t_n + h
        y_np1_i = Y[0]

        NewtonFunc = lambda t, y: NewtonFunc_hvar(t, y, h)
        jacobian = self.getJacobian(t_np1, y_np1_i, NewtonFunc, NewtonFunc(t_np1, y_np1_i))
        new_jacobian = 1

        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            y_np1_ip1 = y_np1_i - np.linalg.solve(jacobian, y_np1_i)
            if self.getNorm(y_np1_ip1)/self.getNorm(y_np1_i) < 1:
                if np.linalg.norm(y_np1_ip1 - Y[0]) < self.tol:
                    return t_np1, y_np1_ip1, h*2
                else:
                    return None, None, h/2
            else:
                if new_jacobian:
                    h /= 2
                    t_np1 = t_n + h
                NewtonFunc = lambda t, y: NewtonFunc_hvar(t, y, h)
                jacobian = self.getJacobian(t_np1, y_np1_i, NewtonFunc, NewtonFunc(t_np1, y_np1_i))
                new_jacobian = 1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")


class BDF2_Newton(BDF_general):
    """The BDF2 method using Newtons method with variable step size as a corrector.
    Built on the BDF_general class.
    """
    def __init__(self, problem):
        """Initiates an instance of BDF_general.

        Args:
            problem (Explicit_Problem): See Assimulo documentation.
            order (int): The order of BDF to use.
        """
        BDF_general.__init__(self, problem, order=2)

    def getNewtonFunc(self, Y, H):
        """The BDF2 function to use in the Newton method. Variable step size.
        The following work by Hiroaki Nishikawa was used for the step sizes:
        https://www.researchgate.net/publication/351082535_Derivation_of_BDF2BDF3_for_Variable_Step_Size

        Args:
            Y ([floats]): 2 previous points.
            H ([float]): 1 previous step size.

        Returns:
            callable: A function of 3 variables: t, y and h.
        """
        h_nm1 = H[0]
        y_n, y_nm1 = Y
        a_np1 = lambda h: (h_nm1+2*h)/(h*(h_nm1 + h))
        a_nm1 = lambda h: h/(h_nm1 * (h + h_nm1))
        a_n = lambda h: - a_np1(h) - a_nm1(h)
        return lambda t, y, h: a_np1(h) * y + a_n(h)*y_n + a_nm1(h)*y_nm1 - self.problem.rhs(t, y)


class BDF3_Newton(BDF_general):
    """The BDF3 method using Newtons method with variable step size as a corrector.
    Built on the BDF_general class.
    """
    def __init__(self, problem):
        """Initiates an instance of BDF_general.

        Args:
            problem (Explicit_Problem): See Assimulo documentation.
            order (int): The order of BDF to use.
        """
        BDF_general.__init__(self, problem, order=3)

    def getNewtonFunc(self, Y, H):
        """The BDF3 function to use in the Newton method. Variable step size.
        The following work by Hiroaki Nishikawa was used for the step sizes:
        https://www.researchgate.net/publication/351082535_Derivation_of_BDF2BDF3_for_Variable_Step_Size

        Args:
            Y ([floats]): 3 previous points.
            H ([floats]): 2 previous step sizes.

        Returns:
            callable: A function of 3 variables: t, y and h.
        """
        h_nm1, h_nm2 = H
        y_n, y_nm1, y_nm2 = Y
        a_n = lambda h: - ((h + h_nm1) * (h + h_nm1 + h_nm2))/(h*h_nm1*(h_nm1+h_nm2))
        a_nm1 = lambda h: (h * (h + h_nm1 + h_nm2))/(h_nm1 * h_nm2 * (h + h_nm1))
        a_nm2 = lambda h: - (h * (h + h_nm1))/(h_nm2 * (h_nm1 + h_nm2) * (h + h_nm1 + h_nm2))
        a_np1 = lambda h: - (a_n(h) + a_nm1(h) + a_nm2(h))
        return lambda t, y, h: a_np1(h)*y + a_n(h)*y_n + a_nm1(h)*y_nm1 + a_nm2(h)*y_nm2 - self.problem.rhs(t, y)


if __name__ == "__main__":
    # When run as main, compares the methods to CVode using the function 'problem_func' as example.
    def problem_func(t, y):
        temp = spring_constant * (1 - 1/np.sqrt(y[0]**2+y[1]**2))
        return np.asarray([y[2], y[3], -y[0]*temp, -y[1]*temp - 1])
    spring_constant = 500

    starting_point = np.array([1, 0, 0, 0])
    problem = Explicit_Problem(problem_func, y0=starting_point)
    t_end = 10

    from assimulo.solvers.sundials import CVode
    problem.name = f"Task 3, k={spring_constant}, CVode"
    solver = CVode(problem)
    solver.simulate(t_end)
    solver.plot()

    problem.name = f"Task 3, k={spring_constant}, BDF2-solver, Newton"
    solver = BDF2_Newton(problem)
    solver.simulate(t_end)
    solver.plot()

    problem.name = f"Task 3, k={spring_constant}, BDF3-solver, Newton"
    solver = BDF3_Newton(problem)
    solver.simulate(t_end)
    solver.plot()
