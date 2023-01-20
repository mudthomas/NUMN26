from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl

import numpy as np
from BDF2_Assimulo import BDF_2

def f(t, y):
    k = 10
    lam = k * (np.sqrt(y[0]**2+y[1]**2) - 1)/np.sqrt(y[0]**2+y[1]**2)
    return np.asarray([y[2], y[3], -y[0]*lam, -y[1]*lam -1])

pend_mod=Explicit_Problem(f, y0=np.array([1,1,1,1]))
pend_mod.name="Task 1"

exp_sim = BDF_2(pend_mod) #Create a BDF solver
t, y = exp_sim.simulate(1)
exp_sim.plot()
mpl.show()