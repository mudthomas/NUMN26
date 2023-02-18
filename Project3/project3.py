from assimulo.problem import Explicit_Problem
from assimulo.explicit_ode import Explicit_ODE
import numpy as np

class Explicit_Problem_2nd(Explicit_Problem):
    def __init__(self, y0, yp0):
    self.t0 = 0
    self.y0 = np.hstack((y0, yp0))


class Explicit_2nd_Order(Explicit_ODE):
    pass


class Newmark(Explicit_2nd_Order):
    pass


class HHT(Explicit_2nd_Order):
    pass
