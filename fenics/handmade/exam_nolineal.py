import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import mshr as ms

"""
This time we want to resolve the problem

    - ∇² w +w² = p     (1)

As always we try with a testfunction 'v' which multiplies all previous equation,
after we integrate over certain domaind Ω
"""

#Of course we start with a simple domain
nx = ny = 100
mesh = fe.UnitSquareMesh(nx, ny)
#It's necesary create a function space where functions can live peacefully
V = fe.FunctionSpace(mesh, 'P', 1)

"""
On the other hand, we need a boundary conditions. Thus we gonna try the method
of manufactured solutions, which consists of take a inicial known 'trial'function
u(x,y) and through to (1) define p(x,y), and take u(x,y) as boundary condition

If we take u(x,y) = x²+2*y², on 2 dimensions, we obtain

    p(x,y) = x^4+4*x^2*y^2+4*y^4-6
    w_D(x,y) = x^2+2*y^2
"""

w_D = fe.Expression('x[0]*x[0] + 2*x[1]*x[1]', degree=2)

#Check boundary points
def boundary(x, on_boundary):
    return on_boundary

# Dirichlet bpundary conditions
bc = fe.DirichletBC(V, w_D, boundary)

""" Here we introduce the 'v' testfunction, and weak formulation"""

w = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Expression('pow(x[0],4)+4*pow(x[0],2)*pow(x[1],2)+4*pow(x[1],4)-6', degree = 4)
#F = (fe.dot(fe.grad(w), fe.grad(v))*fe.dx + w*w*v*fe.dx - p*v*fe.dx
#a, L = fe.lhs(F), fe.rhs(F)
a = fe.dot(fe.grad(w), fe.grad(v))*fe.dx + w*w*v*fe.dx
L = f*v*fe.dx

# Solve
w = fe.Function(V)
fe.solve(a == L, w, bc)

#Plotting solutions and save it
c = fe.plot(w)
plt.colorbar(c)
vtkfile = fe.File('exam_nolineal.pvd')
vtkfile << w
#Error on L2 space
error_L2 = fe.errornorm(w_D, w, 'L2')
print('error_L2  =', error_L2)

plt.show()
