from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import Explicit_Problem, Explicit_ODE_Exception, ID_PY_OK
import numpy as np

import BDF_FPI as FPI


class BDF_general_Newton(FPI.BDF_general):
    """A solver for ODEs using the BDF method with Newton as corrector.
       Implements the Explicit_ODE method from Assimulo.
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
        self.atol = np.ones(len(problem.y0)) * self.tol

    def integrate(self, t0, y0, tf, opts):
        """Solves the ODE problem.
        First it takes one step with the explicit Euler method,
        then it takes steps using the BDF method of increasing order util self.order is reached.
        After self.order number of steps it continues taking steps of that BDF method util reaching tf.

        Args:
            t0 (float-like): The starting time
            y0 (array([floats])): The starting point
            tf (_type_): The end time
            opts (_type_): _description_

        Returns:
            (ID_PY_OK, [floats], [floats]): ID_PY_OK, a list of the evaluation times and a list of the values.
        """
        h = min(self._get_h(), np.abs(tf - t0))
        t_list = [t0]
        y_list = [y0]

        self.statistics["nsteps"] += 1
        t, y = self.EEstep(t0, y0, h)
        t_list.append(t)
        y_list.append(y)
        h = min(self._get_h(), np.abs(tf - t0))

        for i in range(1, self.order):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            t, y = self.BDFstep_general(t, y_list[::-1], h)
            t_list.append(t)
            y_list.append(y)
            h = min(h, np.abs(tf - t))

        for i in range(self.order, self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            t, y = self.BDFstep(t, y_list[-self.order:][::-1], h)
            t_list.append(t)
            y_list.append(y)
            h = min(h, np.abs(tf - t))

        return ID_PY_OK, t_list, y_list

    def BDFstep_Newton(self, t_n, Y, h, coeffs=None):
        if coeffs is None:
            coeffs = self.getBDFcoeffs(len(Y))

        static_terms = coeffs[0] * Y[0]
        for i in range(1, len(Y)):
            static_terms += coeffs[i] * Y[i]

        NewtonFunc = lambda t, y: static_terms + coeffs[-1] * h * self.problem.rhs(t, y) - y

        t_np1 = t_n + h
        y_np1_i = Y[0]

        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            y_np1_ip1 = NewtonFunc(t_np1, y_np1_i)
            jacobian = self.getJacobian(t_np1, y_np1_i, NewtonFunc, y_np1_ip1)
            y_np1_ip1 = Y[0] - np.linalg.solve(jacobian, y_np1_ip1)
            if (np.abs(y_np1_ip1 - y_np1_ip1) < self.atol).all():
                return t_np1, y_np1_ip1
            y_np1_i = y_np1_ip1
        else:
            raise Explicit_ODE_Exception(f"Corrector could not converge within {i} iterations")

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
            jacobian[:, j] = (Newton_func(t_np1, point + shift) - evaluation) / finite_diff
            self.statistics["nfcns"] += 1
        return jacobian

    def BDFstep(self, t_n, Y, h):
        """One step with the BDF method.
        If no coeffs are given, the appropriate coeffs given the length of Y will be used.

        Args:
            t_n (float): Last evaluation time.
            Y ([floats]): Previous values.
            h (float): Step size.
            coeffs ([floats], optional): The coefficients for the BDF-method. Defaults to None.

        Raises:
            Explicit_ODE_Exception: See Assimulo documentation.

        Returns:
            (float, float): The next evaluation time and its value.
        """
        return self.BDFstep_Newton(t_n, Y, h)


class BDF1_Newton(BDF_general_Newton):
    """A solver for ODEs using the BDF method of order 1 with Newton as corrector.
       Implements the BDF_general_Newton method.
       y_np1 - y_n = h*f(t_np1, y_np1)
    """
    def __init__(self, problem):
        """Initiates an instance of BDF1.

        Args:
            problem (Explicit_Problem): See Assimulo documentation.
        """
        BDF_general_Newton.__init__(self, problem, 1)

    def BDFstep(self, t_n, Y, h):
        """One step in the BDF1 method.
        Calls the BDFstep_general method with the correct coefficients.

        Args:
            t_n (float): Previous evaluation time
            Y ([floats]): 1 previous value.
            h (float): Step size.

        Returns:
            (float, float): The next evaluation time and its value.
        """
        return self.BDFstep_Newton(t_n, Y, h, coeffs=[1, 1])


class BDF2_Newton(BDF_general_Newton):
    """A solver for ODEs using the BDF method of order 2 with Newton as corrector.
       Implements the BDF_general_Newton method.
       y_np1 - (4/3)y_n + (1/3)y_nm1 = (2/3)h*f(t_np1, y_np1)
    """
    def __init__(self, problem):
        """Initiates an instance of BDF2.

        Args:
            problem (Explicit_Problem): See Assimulo documentation.
        """
        BDF_general_Newton.__init__(self, problem, 2)

    def BDFstep(self, t_n, Y, h):
        """One step in the BDF2 method.
        Calls the BDFstep_general method with the correct coefficients.

        Args:
            t_n (float): Previous evaluation time
            Y ([floats]): 2 previous values.
            h (float): Step size.

        Returns:
            (float, float): The next evaluation time and its value.
        """
        return self.BDFstep_Newton(t_n, Y, h, coeffs=[4/3, -1/3, 2/3])


class BDF3_Newton(BDF_general_Newton):
    """A solver for ODEs using the BDF method of order 3 with Newton as corrector.
       Implements the BDF_general_Newton method.
       y_np1 - (18/11)y_n + (9/11)y_nm1 - (2/11)y_nm2 = (6/11)h*f(t_np1, y_np1)
    """
    def __init__(self, problem):
        """Initiates an instance of BDF3.

        Args:
            problem (Explicit_Problem): See Assimulo documentation.
        """
        BDF_general_Newton.__init__(self, problem, 3)

    def BDFstep(self, t_n, Y, h):
        """One step in the BDF3 method.
        Calls the BDFstep_general method with the correct coefficients.

        Args:
            t_n (float): Previous evaluation time
            Y ([floats]): 3 previous values.
            h (float): Step size.

        Returns:
            (float, float): The next evaluation time and its value.
        """
        return self.BDFstep_Newton(t_n, Y, h, coeffs=[18/11, -9/11, 2/11, 6/11])


class BDF4_Newton(BDF_general_Newton):
    """A solver for ODEs using the BDF method of order 4 with Newton as corrector.
       Implements the BDF_general_Newton method.
       y_np1 - (48/25)y_n + (36/25)y_nm1 - (16/25)y_nm2 + (3/25)y_nm3 = (12/25)h*f(t_np1, y_np1)
    """
    def __init__(self, problem):
        """Initiates an instance of BDF4.

        Args:
            problem (Explicit_Problem): See Assimulo documentation.
        """
        BDF_general_Newton.__init__(self, problem, 4)

    def BDFstep(self, t_n, Y, h):
        """One step in the BDF4 method.
        Calls the BDFstep_general method with the correct coefficients.

        Args:
            t_n (float): Previous evaluation time
            Y ([floats]): 4 previous values.
            h (float): Step size.

        Returns:
            (float, float): The next evaluation time and its value.
        """
        return self.BDFstep_Newton(t_n, Y, h, coeffs=[48/25, -36/25, 16/25, -3/25, 12/25])


class BDF5_Newton(BDF_general_Newton):
    """A solver for ODEs using the BDF method of order 5 with Newton as corrector.
       Implements the BDF_general_Newton method.
       y_np1 - (300/137)y_n + (300/137)y_nm1 - (200/137)y_nm2 + (75/137)y_nm3
       - (12/137)y_nm4 = (60/137)h*f(t_np1, y_np1)
    """
    def __init__(self, problem):
        """Initiates an instance of BDF5.

        Args:
            problem (Explicit_Problem): See Assimulo documentation.
        """
        BDF_general_Newton.__init__(self, problem, 5)

    def BDFstep(self, t_n, Y, h):
        """One step in the BDF5 method.
        Calls the BDFstep_general method with the correct coefficients.

        Args:
            t_n (float): Previous evaluation time
            Y ([floats]): 5 previous values.
            h (float): Step size.

        Returns:
            (float, float): The next evaluation time and its value.
        """
        return self.BDFstep_Newton(t_n, Y, h, coeffs=[300/137, -300/137, 200/137, -75/137, 12/137, 60/137])


class BDF6_Newton(BDF_general_Newton):
    """A solver for ODEs using the BDF method of order 6 with Newton as corrector.
       Implements the BDF_general_Newton method.
       y_np1 - (360/147)y_n + (450/147)y_nm1 - (400/147)y_nm2 + (225/147)y_nm3
       - (72/147)y_nm4 + (10/147)y_nm5 = (60/147)h*f(t_np1, y_np1)
    """
    def __init__(self, problem):
        """Initiates an instance of BDF6.

        Args:
            problem (Explicit_Problem): See Assimulo documentation.
        """
        BDF_general_Newton.__init__(self, problem, 6)

    def BDFstep(self, t_n, Y, h):
        """One step in the BDF6 method.
        Calls the BDFstep_general method with the correct coefficients.

        Args:
            t_n (float): Previous evaluation time
            Y ([floats]): 6 previous values.
            h (float): Step size.

        Returns:
            (float, float): The next evaluation time and its value.
        """
        return self.BDFstep_Newton(t_n, Y, h, coeffs=[360/147, -450/147, 400/147, -225/147, 72/147, -10/147, 60/147])


if __name__ == "__main__":
    # When run as main, compares the methods to CVode using the function 'problem_func' as example.
    def problem_func(t, y):
        temp = spring_constant * (1 - 1 / np.sqrt(y[0]**2 + y[1]**2))
        return np.asarray([y[2], y[3], -y[0] * temp, -y[1] * temp - 1])
    spring_constant = 3

    starting_point = np.array([1 - 1e-6, 0, 0, 0])
    problem = Explicit_Problem(problem_func, y0=starting_point)
    t_end = 10

    from assimulo.solvers.sundials import CVode
    problem.name = "CVode"
    solver = CVode(problem)
    solver.simulate(t_end)
    solver.plot()

    problem.name = "BDF1-solver, Newton"
    solver = BDF1_Newton(problem)
    solver.simulate(t_end)
    solver.plot()

    problem.name = "BDF2-solver, Newton"
    solver = BDF2_Newton(problem)
    solver.simulate(t_end)
    solver.plot()

    problem.name = "BDF3-solver, Newton"
    solver = BDF3_Newton(problem)
    solver.simulate(t_end)
    solver.plot()

    problem.name = "BDF4-solver, Newton"
    solver = BDF4_Newton(problem)
    solver.simulate(t_end)
    solver.plot()

    problem.name = "BDF5-solver, Newton"
    solver = BDF5_Newton(problem)
    solver.simulate(t_end)
    solver.plot()

    problem.name = "BDF6-solver, Newton"
    solver = BDF6_Newton(problem)
    solver.simulate(t_end)
    solver.plot()
