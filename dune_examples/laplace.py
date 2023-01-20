# <markdowncell>
# # Re-entrant Corner Problem
#
# Here we will consider the classic _re-entrant corner_ problem,
# \begin{align*}
# -\Delta u &= f, && \text{in } \Omega, \\
# u &= g, && \text{on } \partial\Omega,
# \end{align*}
# where the domain is given using polar coordinates,
# \begin{gather*}
# \Omega = \{ (r,\varphi)\colon r\in(0,1), \varphi\in(0,\Phi) \}~.
# \end{gather*}
# For the boundary condition $g$, we set it to the trace of the function $u$, given by
# \begin{gather*}
# u(r,\varphi) = r^{\frac{\pi}{\Phi}} \sin\big(\frac{\pi}{\Phi} \varphi \big)
# \end{gather*}


# <codecell>
import os
# ensure some compilation output for this example
os.environ['DUNE_LOG_LEVEL'] = 'info'
print("Using DUNE_LOG_LEVEL=",os.getenv('DUNE_LOG_LEVEL'))

import matplotlib
matplotlib.rc( 'image', cmap='jet' )
import math
import numpy
import matplotlib.pyplot as pyplot
from dune.fem.plotting import plotPointData as plot
import dune.grid as grid
import dune.fem as fem
from dune.fem.view import adaptiveLeafGridView as adaptiveGridView
from dune.fem.space import lagrange as solutionSpace
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.function import integrate, uflFunction
from ufl import *
from dune.ufl import DirichletBC


# set the angle for the corner (0<angle<=360)
cornerAngle = 320.

# use a second order space
order = 1


# <markdowncell>
# We first define the domain and set up the grid and space.
# We need this twice - once for a computation on a globally refined grid
# and once for an adaptive one so we put the setup into a function:
#
# We first define the grid for this domain (vertices are the origin and 4
# equally spaced points on the unit sphere starting with (1,0) and
# ending at (cos(cornerAngle), sin(cornerAngle))
#
# Next we define the model together with the exact solution.



# <codecell>
def setup():
    vertices = numpy.zeros((8, 2))
    vertices[0] = [0, 0]
    for i in range(0, 7):
        vertices[i+1] = [math.cos(cornerAngle/6*math.pi/180*i),
                         math.sin(cornerAngle/6*math.pi/180*i)]
    triangles = numpy.array([[2,1,0], [0,3,2], [4,3,0],
                             [0,5,4], [6,5,0], [0,7,6]])

    domain = {"vertices": vertices, "simplices": triangles}
    gridView = adaptiveGridView( leafGridView(domain) )
    gridView.hierarchicalGrid.globalRefine(2)
    space = solutionSpace(gridView, order=order, storage="istl")

    from dune.fem.scheme import galerkin as solutionScheme
    u = TrialFunction(space)
    v = TestFunction(space)
    x = SpatialCoordinate(space.cell())

    # exact solution for this angle
    Phi = cornerAngle / 180 * pi
    phi = atan_2(x[1], x[0]) + conditional(x[1] < 0, 2*pi, 0)
    exact = dot(x, x)**(pi/2/Phi) * sin(pi/Phi * phi)
    a = dot(grad(u), grad(v)) * dx

    # set up the scheme
    laplace = solutionScheme([a==0, DirichletBC(space, exact, 1)], solver="cg",
                parameters={"newton.linear.preconditioning.method":"amg-ilu"})
    uh = space.interpolate([0], name="solution")
    return uh, exact, laplace


# <markdowncell>
# We will start with computing the $H^1$ error on a sequence of globally
# refined grids
#
# Note that by using `fem.globalRefine` instead of
# `hierarchicalGrid.globalRefine` we can prolongate discrete functions
# to the next level. The second argument can also be a list/tuple
# of discrete functions to prolong. With this approach we optain a good
# initial guess for solving the problem on the refined grid.



# <codecell>
uh, exact, laplace = setup()
h1error = uflFunction(uh.space.gridView, name="h1error", order=uh.space.order, ufl=dot(grad(uh - exact), grad(uh - exact)))
errorGlobal = []
dofsGlobal  = []
for count in range(3):
    laplace.solve(target=uh)
    error = math.sqrt(integrate(uh.space.gridView, h1error, 5))
    errorGlobal += [error]
    dofsGlobal  += [uh.space.size]
    print(count, ": size=", uh.space.gridView.size(0), "error=", error)
    uh.plot()
    fem.globalRefine(1,uh)


