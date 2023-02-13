from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import IDA
from assimulo.solvers.runge_kutta import RungeKutta4
import numpy as np
import scipy.optimize as opt


class Seven_bar_mechanism_exp(Explicit_Problem):
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
		yp = np.hstack((np.zeros(7,), new))
		return y, yp

	def Newton_init(self, x, y):
		# Inertia data
		m1, m2, m3, m4, m5, m6, m7 = .04325, .00365, .02373, .00706, .07050, .00706, .05498
		i1, i2, i3, i4, i5, i6, i7 = 2.194e-6, 4.410e-7, 5.255e-6, 5.667e-7, 1.169e-5, 5.667e-7, 1.912e-5

		d, da, e, ea = 28.e-3, 115.e-4, 2.e-2, 1421.e-5
		rr, ra = 7.e-3, 92.e-5
		ss, sa, sb = 35.e-3, 1874.e-5, 1043.e-5
		ta, tb = 2308.e-5, 916.e-5
		u, ua, ub = 4.e-2, 1228.e-5, 449.e-5
		zf, zt = 2.e-2, 4.e-2
		fa = 1421.e-5

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

	def rhs(self, t, y):
		"""
		Residual function of the 7-bar mechanism in
		Hairer, Vol. II, p. 533 ff, see also formula (7.11)
		written in residual form
		y,yp vector of dim 20, t scalar
		"""
		# Inertia data
		m1, m2, m3, m4, m5, m6, m7 = .04325, .00365, .02373, .00706, .07050, .00706, .05498
		i1, i2, i3, i4, i5, i6, i7 = 2.194e-6, 4.410e-7, 5.255e-6, 5.667e-7, 1.169e-5, 5.667e-7, 1.912e-5
		# Geometry
		xa, ya = -.06934, -.00227
		xb, yb = -.03635, .03273
		xc, yc = .014, .072

		d, da, e, ea = 28.e-3, 115.e-4, 2.e-2, 1421.e-5
		rr, ra = 7.e-3, 92.e-5
		ss, sa, sb, sc, sd = 35.e-3, 1874.e-5, 1043.e-5, 18.e-3, 2.e-2
		ta, tb = 2308.e-5, 916.e-5
		u, ua, ub = 4.e-2, 1228.e-5, 449.e-5
		zf, zt = 2.e-2, 4.e-2
		fa = 1421.e-5
		# Driving torque
		mom = 0.033
		# Spring data
		c0 = 4530.
		lo = 0.07785

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

		#   Applied forces
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

		#  constraint matrix  G
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

		#     Construction of the residual
		res_1 = y[7:14]

		v1, v2, v3, v4, v5, v6, v7 = bep, thp, gap, php, dep, omp, epp
		gqq = np.array([
			- rr * cobe * v1**2 + d * cobeth * (v1 + v2)**2 + ss * siga * v3**2,
			- rr * sibe * v1**2 + d * sibeth * (v1 + v2)**2 - ss * coga * v3**2,
			- rr * cobe * v1**2 + d * cobeth * (v1 + v2)**2 + e * siphde * (v4 + v5)**2  + zt * code * v5**2,
			- rr * sibe * v1 ** 2 + d * sibeth * (v1 + v2)**2 - e * cophde * (v4 + v5)**2 + zt * side * v5**2,
			- rr * cobe * v1**2 + d * cobeth * (v1 + v2)**2 + zf * coomep * (v6 + v7)**2 + u * siep * v7**2,
			- rr * sibe * v1**2 + d * sibeth * (v1 + v2)**2 + zf * siomep * (v6 + v7)**2 - u * coep * v7**2])

		res_2 = np.linalg.solve(np.vstack((np.hstack((m, gp.T)), np.hstack((gp, np.zeros((6, 6)))))), np.hstack((ff[0:7], -gqq)))

		return np.hstack((res_1, res_2))


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	def doExp():
		problem = Seven_bar_mechanism_exp()

		solver = RungeKutta4(problem)
		solver._set_h(0.0001)

		# tol1 = 1.e-6 * np.ones(20)
		# solver.atol = tol1
		t1, y1 = solver.simulate(0.03)
		# solver.plot()
		plt.plot(t1, np.fmod(y1[:, 0:7], 2 * np.pi))
		plt.ylim(-1, 1)
		plt.show()

	doExp()
