from assimulo.problem import Implicit_Problem, Explicit_Problem
from assimulo.solvers.sundials import IDA
from assimulo.solvers.runge_kutta import RungeKutta4
import numpy as np
import scipy.optimize as opt

from input_data import (
	m1, m2, m3, m4, m5, m6, m7,
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

	def __init__(self, Newton=False):
		self.Newton = Newton
		self.y0, self.yd0 = self.init_squeezer()
		self.t0 = 0
		self.algvar = np.ones(len(self.y0))

	def reset(self):
		self.y0, self.yd0 = self.init_squeezer()

	def init_squeezer(self):
		y = np.hstack((np.array([
			-0.0617138900142764496358948458001,  # beta
			0.,                                  # theta
			0.455279819163070380255912382449,    # gamma
			0.222668390165885884674473185609,    # phi
			0.487364979543842550225598953530,    # delta
			-0.222668390165885884674473185609,   # Omega
			1.23054744454982119249735015568]),   # epsilon
			np.zeros((7,)), np.array([
				98.5668703962410896057654982170,     # lambda[0]
				-6.12268834425566265503114393122]),   # lambda[1]
			np.zeros((4,))))

		if self.Newton:
			new = opt.fsolve(self.Newton_init, 1e-6 * np.ones(13,), args=y)
			return y, np.hstack((new[:7], np.array([
				14222.4439199541138705911625887,        # betadotdot
				-10666.8329399655854029433719415,       # Thetadotdot
				0., 0., 0., 0., 0.]), new[7:]))
		else:
			return y, np.hstack((np.zeros(7,), np.array([
				14222.4439199541138705911625887,        # betadotdot
				-10666.8329399655854029433719415,       # Thetadotdot
				0., 0., 0., 0., 0.]), np.zeros((6,))))

	def Newton_init(self, x, y):
		# Initial computations and assignments
		beta, theta, gamma, phi, delta, omega, epsilon = y[0:7]
		bep, thp, gap, php, dep, omp, epp = y[7:14]
		sibe, sith, siga, siph, side, siom, siep = np.sin(y[0:7])
		cobe, coth, coga, coph, code, coom, coep = np.cos(y[0:7])
		sibeth, cobeth = np.sin(beta + theta), np.cos(beta + theta)
		siphde, cophde = np.sin(phi + delta), np.cos(phi + delta)
		siomep, coomep = np.sin(omega + epsilon), np.cos(omega + epsilon)

		m = np.zeros((7, 7))
		m[0, 0] = m1 * ra**2 + m2 * (rr**2 - 2 * da * rr * coth + da**2) + i1 + i2
		m[1, 0] = m[0, 1] = m2 * (da**2 - da * rr * coth) + i2
		m[1, 1] = m2 * da**2 + i2
		m[2, 2] = m3 * (sa**2 + sb**2) + i3
		m[3, 3] = m4 * (e - ea)**2 + i4
		m[4, 3] = m[3, 4] = m4 * ((e - ea)**2 + zt * (e - ea) * siph) + i4
		m[4, 4] = m4 * (zt**2 + 2 * zt * (e - ea) * siph + (e - ea)**2) + m5 * (ta**2 + tb**2) + i4 + i5
		m[5, 5] = m6 * (zf - fa)**2 + i6
		m[6, 5] = m[5, 6] = m6 * ((zf - fa)**2 - u * (zf - fa) * siom) + i6
		m[6, 6] = m6 * ((zf - fa)**2 - 2 * u * (zf - fa) * siom + u**2) + m7 * (ua**2 + ub**2) + i6 + i7

		gp = np.zeros((6, 7))
		gp[0, 0] = - rr * sibe + d * sibeth
		gp[0, 1] = d * sibeth
		gp[0, 2] = - ss * coga
		gp[1, 0] = rr * cobe - d * cobeth
		gp[1, 1] = - d * cobeth
		gp[1, 2] = - ss * siga
		gp[2, 0] = - rr * sibe + d * sibeth
		gp[2, 1] = d * sibeth
		gp[2, 3] = - e * cophde
		gp[2, 4] = - e * cophde + zt * side
		gp[3, 0] = rr * cobe - d * cobeth
		gp[3, 1] = - d * cobeth
		gp[3, 3] = - e * siphde
		gp[3, 4] = - e * siphde - zt * code
		gp[4, 0] = - rr * sibe + d * sibeth
		gp[4, 1] = d * sibeth
		gp[4, 5] = zf * siomep
		gp[4, 6] = zf * siomep - u * coep
		gp[5, 0] = rr * cobe - d * cobeth
		gp[5, 1] = - d * cobeth
		gp[5, 5] = - zf * coomep
		gp[5, 6] = - zf * coomep - u * siep

		return np.vstack((np.hstack((m, gp.T)), np.hstack((gp, np.zeros((6, 6)))))) @ x

	def res(self, t, y, yp):
		"""
		Residual function of the 7-bar mechanism in
		Hairer, Vol. II, p. 533 ff, see also formula (7.11)
		written in residual form
		y,yp vector of dim 20, t scalar
		"""

		# Initial computations and assignments
		beta, theta, gamma, phi, delta, omega, epsilon = y[0:7]
		bep, thp, gap, php, dep, omp, epp = y[7:14]
		lamb = y[14:20]
		sibe, sith, siga, siph, side, siom, siep = np.sin(y[0:7])
		cobe, coth, coga, coph, code, coom, coep = np.cos(y[0:7])
		sibeth, cobeth = np.sin(beta + theta), np.cos(beta + theta)
		siphde, cophde = np.sin(phi + delta), np.cos(phi + delta)
		siomep, coomep = np.sin(omega + epsilon), np.cos(omega + epsilon)

		# Mass matrix
		m = np.zeros((7, 7))
		m[0, 0] = m1 * ra**2 + m2 * (rr**2 - 2 * da * rr * coth + da**2) + i1 + i2
		m[1, 0] = m[0, 1] = m2 * (da**2 - da * rr * coth) + i2
		m[1, 1] = m2 * da**2 + i2
		m[2, 2] = m3 * (sa**2 + sb**2) + i3
		m[3, 3] = m4 * (e - ea)**2 + i4
		m[4, 3] = m[3, 4] = m4 * ((e - ea)**2 + zt * (e - ea) * siph) + i4
		m[4, 4] = m4 * (zt**2 + 2 * zt * (e - ea) * siph + (e - ea)**2) + m5 * (ta**2 + tb**2) + i4 + i5
		m[5, 5] = m6 * (zf - fa)**2 + i6
		m[6, 5] = m[5, 6] = m6 * ((zf - fa)**2 - u * (zf - fa) * siom) + i6
		m[6, 6] = m6 * ((zf - fa)**2 - 2 * u * (zf - fa) * siom + u**2) + m7 * (ua**2 + ub**2) + i6 + i7

		# Applied forces
		xd = sd * coga + sc * siga + xb
		yd = sd * siga - sc * coga + yb
		lang = np.sqrt((xd - xc)**2 + (yd - yc)**2)
		force = - c0 * (lang - lo) / lang
		fx = force * (xd - xc)
		fy = force * (yd - yc)
		ff = np.array([
			mom - m2 * da * rr * thp * (thp + 2 * bep) * sith,
			m2 * da * rr * bep**2 * sith,
			fx * (sc * coga - sd * siga) + fy * (sd * coga + sc * siga),
			m4 * zt * (e - ea) * dep**2 * coph,
			- m4 * zt * (e - ea) * php * (php + 2 * dep) * coph,
			- m6 * u * (zf - fa) * epp**2 * coom,
			m6 * u * (zf - fa) * omp * (omp + 2 * epp) * coom])

		# Constraint matrix, G
		gp = np.zeros((6, 7))
		gp[0, 0] = - rr * sibe + d * sibeth
		gp[0, 1] = d * sibeth
		gp[0, 2] = - ss * coga
		gp[1, 0] = rr * cobe - d * cobeth
		gp[1, 1] = - d * cobeth
		gp[1, 2] = - ss * siga
		gp[2, 0] = - rr * sibe + d * sibeth
		gp[2, 1] = d * sibeth
		gp[2, 3] = - e * cophde
		gp[2, 4] = - e * cophde + zt * side
		gp[3, 0] = rr * cobe - d * cobeth
		gp[3, 1] = - d * cobeth
		gp[3, 3] = - e * siphde
		gp[3, 4] = - e * siphde - zt * code
		gp[4, 0] = - rr * sibe + d * sibeth
		gp[4, 1] = d * sibeth
		gp[4, 5] = zf * siomep
		gp[4, 6] = zf * siomep - u * coep
		gp[5, 0] = rr * cobe - d * cobeth
		gp[5, 1] = - d * cobeth
		gp[5, 5] = - zf * coomep
		gp[5, 6] = - zf * coomep - u * siep

		# Construction of the residual
		if self.index == 3:  # Index-3 constraint
			res_3 = np.array([
				rr * cobe - d * cobeth - ss * siga - xb,
				rr * sibe - d * sibeth + ss * coga - yb,
				rr * cobe - d * cobeth - e * siphde - zt * code - xa,
				rr * sibe - d * sibeth + e * cophde - zt * side - ya,
				rr * cobe - d * cobeth - zf * coomep - u * siep - xa,
				rr * sibe - d * sibeth - zf * siomep + u * coep - ya])
		if self.index == 2:  # Index-2 constraint
			res_3 = np.dot(gp, y[7:14])
		if self.index == 1:  # Index-1 constraint
			res_3 = np.array([
				- rr * cobe * bep**2 + d * cobeth * (bep + thp)**2 + ss * siga * gap**2,
				- rr * sibe * bep**2 + d * sibeth * (bep + thp)**2 - ss * coga * gap**2,
				- rr * cobe * bep**2 + d * cobeth * (bep + thp)**2 + e * siphde * (php + dep)**2 + zt * code * dep**2,
				- rr * sibe * bep ** 2 + d * sibeth * (bep + thp)**2 - e * cophde * (php + dep)**2 + zt * side * dep**2,
				- rr * cobe * bep**2 + d * cobeth * (bep + thp)**2 + zf * coomep * (omp + epp)**2 + u * siep * epp**2,
				- rr * sibe * bep**2 + d * sibeth * (bep + thp)**2 + zf * siomep * (omp + epp)**2 - u * coep * epp**2])
			res_3 += np.dot(gp, yp[7:14])

		return np.hstack((yp[0:7] - y[7:14], np.dot(m, yp[7:14]) - ff[0:7] + np.dot(gp.T, lamb), res_3))

	def rhs(self, t, y):
		"""
		Residual function of the 7-bar mechanism in
		Hairer, Vol. II, p. 533 ff, see also formula (7.11)
		written in residual form
		y,yp vector of dim 20, t scalar
		"""

		# Initial computations and assignments
		beta, theta, gamma, phi, delta, omega, epsilon = y[0:7]
		bep, thp, gap, php, dep, omp, epp = y[7:14]
		sibe, sith, siga, siph, side, siom, siep = np.sin(y[0:7])
		cobe, coth, coga, coph, code, coom, coep = np.cos(y[0:7])
		sibeth, cobeth = np.sin(beta + theta), np.cos(beta + theta)
		siphde, cophde = np.sin(phi + delta), np.cos(phi + delta)
		siomep, coomep = np.sin(omega + epsilon), np.cos(omega + epsilon)

		# Mass matrix
		m = np.zeros((7, 7))
		m[0, 0] = m1 * ra**2 + m2 * (rr**2 - 2 * da * rr * coth + da**2) + i1 + i2
		m[1, 0] = m[0, 1] = m2 * (da**2 - da * rr * coth) + i2
		m[1, 1] = m2 * da**2 + i2
		m[2, 2] = m3 * (sa**2 + sb**2) + i3
		m[3, 3] = m4 * (e - ea)**2 + i4
		m[4, 3] = m[3, 4] = m4 * ((e - ea)**2 + zt * (e - ea) * siph) + i4
		m[4, 4] = m4 * (zt**2 + 2 * zt * (e - ea) * siph + (e - ea)**2) + m5 * (ta**2 + tb**2) + i4 + i5
		m[5, 5] = m6 * (zf - fa)**2 + i6
		m[6, 5] = m[5, 6] = m6 * ((zf - fa)**2 - u * (zf - fa) * siom) + i6
		m[6, 6] = m6 * ((zf - fa)**2 - 2 * u * (zf - fa) * siom + u**2) + m7 * (ua**2 + ub**2) + i6 + i7

		# Applied forces
		xd = sd * coga + sc * siga + xb
		yd = sd * siga - sc * coga + yb
		lang = np.sqrt((xd - xc)**2 + (yd - yc)**2)
		force = - c0 * (lang - lo) / lang
		fx = force * (xd - xc)
		fy = force * (yd - yc)
		ff = np.array([
			mom - m2 * da * rr * thp * (thp + 2 * bep) * sith,
			m2 * da * rr * bep**2 * sith,
			fx * (sc * coga - sd * siga) + fy * (sd * coga + sc * siga),
			m4 * zt * (e - ea) * dep**2 * coph,
			- m4 * zt * (e - ea) * php * (php + 2 * dep) * coph,
			- m6 * u * (zf - fa) * epp**2 * coom,
			m6 * u * (zf - fa) * omp * (omp + 2 * epp) * coom])

		# Constraint matrix, G
		gp = np.zeros((6, 7))
		gp[0, 0] = - rr * sibe + d * sibeth
		gp[0, 1] = d * sibeth
		gp[0, 2] = - ss * coga
		gp[1, 0] = rr * cobe - d * cobeth
		gp[1, 1] = - d * cobeth
		gp[1, 2] = - ss * siga
		gp[2, 0] = - rr * sibe + d * sibeth
		gp[2, 1] = d * sibeth
		gp[2, 3] = - e * cophde
		gp[2, 4] = - e * cophde + zt * side
		gp[3, 0] = rr * cobe - d * cobeth
		gp[3, 1] = - d * cobeth
		gp[3, 3] = - e * siphde
		gp[3, 4] = - e * siphde - zt * code
		gp[4, 0] = - rr * sibe + d * sibeth
		gp[4, 1] = d * sibeth
		gp[4, 5] = zf * siomep
		gp[4, 6] = zf * siomep - u * coep
		gp[5, 0] = rr * cobe - d * cobeth
		gp[5, 1] = - d * cobeth
		gp[5, 5] = - zf * coomep
		gp[5, 6] = - zf * coomep - u * siep

		gqq = np.array([
			- rr * cobe * bep**2 + d * cobeth * (bep + thp)**2 + ss * siga * gap**2,
			- rr * sibe * bep**2 + d * sibeth * (bep + thp)**2 - ss * coga * gap**2,
			- rr * cobe * bep**2 + d * cobeth * (bep + thp)**2 + e * siphde * (php + dep)**2 + zt * code * dep**2,
			- rr * sibe * bep ** 2 + d * sibeth * (bep + thp)**2 - e * cophde * (php + dep)**2 + zt * side * dep**2,
			- rr * cobe * bep**2 + d * cobeth * (bep + thp)**2 + zf * coomep * (omp + epp)**2 + u * siep * epp**2,
			- rr * sibe * bep**2 + d * sibeth * (bep + thp)**2 + zf * siomep * (omp + epp)**2 - u * coep * epp**2])

		res_2 = np.linalg.solve(np.vstack((np.hstack((m, gp.T)), np.hstack((gp, np.zeros((6, 6)))))), np.hstack((ff, -gqq)))

		return np.hstack((y[7:14], res_2))


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
		Seven_bar_mechanism.__init__(self, Newton=True)
		self.index = 1


class Seven_bar_mechanism_exp(Explicit_Problem):
	def __init__(self):
		Seven_bar_mechanism.__init__(self)

	def init_squeezer(self):
		return Seven_bar_mechanism.init_squeezer(self)

	def Newton_init(self, x, y):
		return Seven_bar_mechanism.Newton_init(self, x, y)

	def rhs(self, t, y):
		return Seven_bar_mechanism.rhs(self, t, y)


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
		tol3 = 1.e-6 * np.ones(20)
		tol3[7:] = 1.e5 * np.ones(13)
		solver.atol = tol3
		t, y, _ = solver.simulate(0.03)
		reportPlot(t, y, "Index-3, IDA solver")

	def doIndex2():
		problem = Seven_bar_mechanism_index2()
		solver = IDA(problem)
		tol2 = 1.e-6 * np.ones(20)
		tol2[14:] = 1.e5 * np.ones(6)
		solver.atol = tol2
		t, y, _ = solver.simulate(0.03)
		reportPlot(t, y, "Index-2, IDA solver")

	def doIndex1():
		problem = Seven_bar_mechanism_index1()
		solver = IDA(problem)
		tol1 = 1.e-6 * np.ones(20)
		solver.atol = tol1
		t, y, _ = solver.simulate(0.03)
		reportPlot(t, y, "Index-1, IDA solver")

	def doExp():
		problem = Seven_bar_mechanism_exp()
		solver = RungeKutta4(problem)
		solver._set_h(0.0001)
		t, y = solver.simulate(0.03)
		reportPlot(t, y, "Explicit, RK4")

	doIndex3()
	doIndex2()
	doIndex1()
	doExp()
