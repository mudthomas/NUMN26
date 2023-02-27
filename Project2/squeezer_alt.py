from assimulo.problem import Implicit_Problem, Explicit_Problem
from assimulo.solvers.sundials import IDA
from assimulo.solvers.radau5 import Radau5DAE
from assimulo.solvers.runge_kutta import RungeKutta4
import numpy as np
import scipy.optimize as opt

from input_data import (m1, m2, m3, m4, m5, m6, m7,
						i1, i2, i3, i4, i5, i6, i7,
						xa, xb, xc, ya, yb, yc,
						d, da, e, ea, rr, ra,
						ss, sa, sb, sc, sd,
						ta, tb, u, ua, ub,
						zf, zt, fa, mom, c0, lo)


class Seven_bar_mechanism(Implicit_Problem):
	"""
	A class which describes the squeezer according to
	Hairer, Vol. II, p. 533 ff, see also formula (7.11)
	"""
	problem_name = 'Woodpecker w/o friction'

	def __init__(self):
		self.y0, self.yd0 = self.init_squeezer()
		self.t0 = 0
		self.algvar = np.ones(len(self.y0))

	def reset(self):
		self.y0, self.yd0 = self.init_squeezer()

	def init_squeezer(self):
		y_1 = np.array([
			-0.0617138900142764496358948458001,  # beta
			0.,                                  # theta
			0.455279819163070380255912382449,    # gamma
			0.222668390165885884674473185609,    # phi
			0.487364979543842550225598953530,    # delta
			-0.222668390165885884674473185609,   # Omega
			1.23054744454982119249735015568])    # epsilon

		lamb = np.array([
			98.5668703962410896057654982170,     # lambda[0]
			-6.12268834425566265503114393122])   # lambda[1]

		y = np.hstack((y_1, np.zeros((7,)), lamb, np.zeros((4,))))
		yp = np.hstack((np.zeros(7,), np.array([
			14222.4439199541138705911625887,        # betadotdot
			-10666.8329399655854029433719415,       # Thetadotdot
			0., 0., 0., 0., 0.]), np.zeros((6,))))

		new = opt.fsolve(self.Newton_init, 1e-6*np.ones(13,), args=y)
		yp = np.hstack((new[:7], np.array([
			14222.4439199541138705911625887,        # betadotdot
			-10666.8329399655854029433719415,       # Thetadotdot
			0., 0., 0., 0., 0.]), new[7:]))
		return y, yp

	def Newton_init(self, x, y):
		self.makeAssignments(y)	  # Initial computations and assignments
		m = self.getMassMatrix()  # Mass matrix
		gp = self.getGP()		  # Constraint matrix, G
		return np.vstack((np.hstack((m, gp.T)), np.hstack((gp, np.zeros((6, 6)))))) @ x

	def makeAssignments(self, y):
		beta, theta, gamma, phi, delta, omega, epsilon = y[0:7]
		self.bep, self.thp, self.gap, self.php, self.dep, self.omp, self.epp = y[7:14]
		self.lamb = y[14:20]
		self.sibe, self.sith, self.siga, self.siph, self.side, self.siom, self.siep = np.sin(y[0:7])
		self.cobe, self.coth, self.coga, self.coph, self.code, self.coom, self.coep = np.cos(y[0:7])
		self.sibeth, self.cobeth = np.sin(beta + theta), np.cos(beta + theta)
		self.siphde, self.cophde = np.sin(phi + delta), np.cos(phi + delta)
		self.siomep, self.coomep = np.sin(omega + epsilon), np.cos(omega + epsilon)

	def getMassMatrix(self):
		m = np.zeros((7, 7))
		m[0, 0] = m1 * ra**2 + m2 * (rr**2 - 2 * da * rr * self.coth + da**2) + i1 + i2
		m[1, 0] = m[0, 1] = m2 * (da**2 - da * rr * self.coth) + i2
		m[1, 1] = m2 * da**2 + i2
		m[2, 2] = m3 * (sa**2 + sb**2) + i3
		m[3, 3] = m4 * (e - ea)**2 + i4
		m[4, 3] = m[3, 4] = m4 * ((e - ea)**2 + zt * (e - ea) * self.siph) + i4
		m[4, 4] = m4 * (zt**2 + 2 * zt * (e - ea) * self.siph + (e - ea)**2) + m5 * (ta**2 + tb**2) + i4 + i5
		m[5, 5] = m6 * (zf - fa)**2 + i6
		m[6, 5] = m[5, 6] = m6 * ((zf - fa)**2 - u * (zf - fa) * self.siom) + i6
		m[6, 6] = m6 * ((zf - fa)**2 - 2 * u * (zf - fa) * self.siom + u**2) + m7 * (ua**2 + ub**2) + i6 + i7
		return m

	def getGP(self):
		gp = np.zeros((6, 7))
		gp[0, 0] = - rr * self.sibe + d * self.sibeth
		gp[0, 1] = d * self.sibeth
		gp[0, 2] = - ss * self.coga
		gp[1, 0] = rr * self.cobe - d * self.cobeth
		gp[1, 1] = - d * self.cobeth
		gp[1, 2] = - ss * self.siga
		gp[2, 0] = - rr * self.sibe + d * self.sibeth
		gp[2, 1] = d * self.sibeth
		gp[2, 3] = - e * self.cophde
		gp[2, 4] = - e * self.cophde + zt * self.side
		gp[3, 0] = rr * self.cobe - d * self.cobeth
		gp[3, 1] = - d * self.cobeth
		gp[3, 3] = - e * self.siphde
		gp[3, 4] = - e * self.siphde - zt * self.code
		gp[4, 0] = - rr * self.sibe + d * self.sibeth
		gp[4, 1] = d * self.sibeth
		gp[4, 5] = zf * self.siomep
		gp[4, 6] = zf * self.siomep - u * self.coep
		gp[5, 0] = rr * self.cobe - d * self.cobeth
		gp[5, 1] = - d * self.cobeth
		gp[5, 5] = - zf * self.coomep
		gp[5, 6] = - zf * self.coomep - u * self.siep
		return gp

	def getFF(self):
		xd = sd * self.coga + sc * self.siga + xb
		yd = sd * self.siga - sc * self.coga + yb
		lang = np.sqrt((xd - xc)**2 + (yd - yc)**2)
		force = - c0 * (lang - lo) / lang
		fx = force * (xd - xc)
		fy = force * (yd - yc)
		return np.array([
			mom - m2 * da * rr * self.thp * (self.thp + 2 * self.bep) * self.sith,
			m2 * da * rr * self.bep**2 * self.sith,
			fx * (sc * self.coga - sd * self.siga) + fy * (sd * self.coga + sc * self.siga),
			m4 * zt * (e - ea) * self.dep**2 * self.coph,
			- m4 * zt * (e - ea) * self.php * (self.php + 2 * self.dep) * self.coph,
			- m6 * u * (zf - fa) * self.epp**2 * self.coom,
			m6 * u * (zf - fa) * self.omp * (self.omp + 2 * self.epp) * self.coom])

	def res(self, t, y, yp):
		"""
		Residual function of the 7-bar mechanism in
		Hairer, Vol. II, p. 533 ff, see also formula (7.11)
		written in residual form
		y,yp vector of dim 20, t scalar
		"""

		self.makeAssignments(y)	  # Initial computations and assignments
		m = self.getMassMatrix()  # Mass matrix
		ff = self.getFF()		  # Applied forces
		gp = self.getGP()		  # Constraint matrix, G

		# Construction of the residual
		if self.index == 3:  # Index-3 constraint
			res_3 = np.array([
				rr * self.cobe - d * self.cobeth - ss * self.siga - xb,
				rr * self.sibe - d * self.sibeth + ss * self.coga - yb,
				rr * self.cobe - d * self.cobeth - e * self.siphde - zt * self.code - xa,
				rr * self.sibe - d * self.sibeth + e * self.cophde - zt * self.side - ya,
				rr * self.cobe - d * self.cobeth - zf * self.coomep - u * self.siep - xa,
				rr * self.sibe - d * self.sibeth - zf * self.siomep + u * self.coep - ya])
		if self.index == 2:  # Index-2 constraint
			res_3 = np.dot(gp, y[7:14])
		if self.index == 1:  # Index-1 constraint
			res_3 = self.getGqq() + np.dot(gp, yp[7:14])

		return np.hstack((yp[0:7] - y[7:14], np.dot(m, yp[7:14]) - ff[0:7] + np.dot(gp.T, self.lamb), res_3))

	def getGqq(self):
		return np.array([
				- rr * self.cobe * self.bep**2 + d * self.cobeth * (self.bep + self.thp)**2 + ss * self.siga * self.gap**2,
				- rr * self.sibe * self.bep**2 + d * self.sibeth * (self.bep + self.thp)**2 - ss * self.coga * self.gap**2,
				- rr * self.cobe * self.bep**2 + d * self.cobeth * (self.bep + self.thp)**2 + e * self.siphde * (self.php + self.dep)**2  + zt * self.code * self.dep**2,
				- rr * self.sibe * self.bep ** 2 + d * self.sibeth * (self.bep + self.thp)**2 - e * self.cophde * (self.php + self.dep)**2 + zt * self.side * self.dep**2,
				- rr * self.cobe * self.bep**2 + d * self.cobeth * (self.bep + self.thp)**2 + zf * self.coomep * (self.omp + self.epp)**2 + u * self.siep * self.epp**2,
				- rr * self.sibe * self.bep**2 + d * self.sibeth * (self.bep + self.thp)**2 + zf * self.siomep * (self.omp + self.epp)**2 - u * self.coep * self.epp**2])

	def rhs(self, t, y):
		"""
		Residual function of the 7-bar mechanism in
		Hairer, Vol. II, p. 533 ff, see also formula (7.11)
		written in residual form
		y,yp vector of dim 20, t scalar
		"""

		self.makeAssignments(y)   # Initial computations and assignments
		m = self.getMassMatrix()  # Mass matrix
		ff = self.getFF()		  # Applied forces
		gp = self.getGP()		  # Constraint matrix, G
		gqq = self.getGqq()

		return np.hstack((y[7:14],
						  np.linalg.solve(np.vstack((np.hstack((m, gp.T)), np.hstack((gp, np.zeros((6, 6)))))), np.hstack((ff, -gqq)))))


class Seven_bar_mechanism_index3(Seven_bar_mechanism):
	def __init__(self):
		Seven_bar_mechanism.__init__(self)
		self.algvar[7:] = np.zeros(13)
		self.index = 3


class Seven_bar_mechanism_index2(Seven_bar_mechanism):
	def __init__(self):
		Seven_bar_mechanism.__init__(self)
		self.algvar[14:] = np.zeros(6)
		self.index = 2


class Seven_bar_mechanism_index1(Seven_bar_mechanism):
	def __init__(self):
		Seven_bar_mechanism.__init__(self)
		self.index = 1


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	def reportPlot(t, y, plottitle):
		plotlabels1 = ["$\\beta$", "$\\Theta$", "$\\gamma$", "$\\Phi$", "$\\delta$", "$\\Omega$", "$\\epsilon$"]
		plotlabels2 = ["$\\dot{\\beta}$", "$\\dot{\\Theta}$", "$\\dot{\\gamma}$", "$\\dot{\\Phi}$", "$\\dot{\\delta}$", "$\\dot{\\Omega}$", "$\\dot{\\epsilon}$"]
		plotlabels3 = ["$\\lambda_1$", "$\\lambda_2$", "$\\lambda_3$", "$\\lambda_4$", "$\\lambda_5$", "$\\lambda_6$"]

		plt.plot(t, np.fmod(y[:, 0:7], 2 * np.pi), label=plotlabels1)
		plt.ylim(-2, 2)
		plt.title(plottitle)
		plt.grid()
		plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
		plt.show()

		plt.plot(t, y[:, 7:14], label=plotlabels2)
		plt.title(plottitle)
		plt.grid()
		plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
		plt.show()

		plt.plot(t, y[:, 14:], label=plotlabels3)
		plt.title(plottitle)
		plt.grid()
		plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
		plt.show()

	def doIndex3():
		problem = Seven_bar_mechanism_index3()
		solver = IDA(problem)
		# solver = Radau5DAE(problem)
		tol3 = 1.e-6 * np.ones(20)
		tol3[7:] = 1.e5 * np.ones(13)
		solver.atol = tol3
		t, y, _ = solver.simulate(0.03)
		reportPlot(t, y, "Index-3, IDA solver")

	def doIndex2():
		problem = Seven_bar_mechanism_index2()
		solver = IDA(problem)
		# solver = Radau5DAE(problem)
		tol2 = 1.e-6 * np.ones(20)
		tol2[14:] = 1.e5 * np.ones(6)
		solver.atol = tol2
		t, y, _ = solver.simulate(0.03)
		reportPlot(t, y, "Index-2, IDA solver")

	def doIndex1():
		problem = Seven_bar_mechanism_index1()
		solver = IDA(problem)
		# solver = Radau5DAE(problem)
		tol1 = 1.e-6 * np.ones(20)
		solver.atol = tol1
		t, y, _ = solver.simulate(0.03)
		reportPlot(t, y, "Index-1, IDA solver")

	def doExp():
		problem = Seven_bar_mechanism()
		solver = RungeKutta4(problem)
		solver._set_h(0.0001)

		t, y = solver.simulate(0.03)
		reportPlot(t, y, "Explicit, RK4")

	doIndex3()
	doIndex2()
	doIndex1()
	# doExp()
