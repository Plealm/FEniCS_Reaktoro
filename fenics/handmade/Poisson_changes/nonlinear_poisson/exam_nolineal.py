import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

"""
This time we want to resolve the problem

    - ∇² w +w² = p     (1)

As always we try with a testfunction 'v' which multiplies all previous equation,
after we integrate over certain domaind Ω
"""

# definición de los simbolos usados en FEniCS
x, y = sym.symbols('x[0], x[1]')
# Definición de la frontera
w = x**2 + 2*y**2
# Se usa Diff de sympy para recrear la ecuación
p = - sym.diff(sym.diff(w, x), x) - sym.diff(sym.diff(w, y), y) + w**2
p = sym.simplify(p)
# formato para escribir las expresiones de C/C++  a Fenics
u_code = sym.printing.ccode(w)
f_code = sym.printing.ccode(p)
print('u =', u_code)
print('f =', f_code)

#Of course we start with a simple domain
nx = ny = 100
mesh = fe.UnitSquareMesh(nx, ny)
#It's necesary create a function space where functions can live peacefully
V = fe.FunctionSpace(mesh, 'P', 1)

"""
On the other hand, we need a boundary conditions. Thus we gonna try the method
of manufactured solutions, which consists of take a inicial known 'trial'function
u(x,y) and through to (1) define p(x,y), and take u(x,y) as boundary condition

If we take w(x,y) = x²+2*y², on 2 dimensions, we obtain

    p(x,y) = x^4+4*x^2*y^2+4*y^4-6
    w_D(x,y) = x^2+2*y^2
"""

w_D = fe.Expression(u_code, degree=2)

#Check boundary points
def boundary(x, on_boundary):
    return on_boundary

# Dirichlet bpundary conditions
bc = fe.DirichletBC(V, w_D, boundary)

""" Here we introduce the 'v' testfunction, and weak formulation"""

w = fe.Function(V) #not TrialFunction
v = fe.TestFunction(V)
f = fe.Expression(f_code, degree = 4)
F = fe.dot(fe.grad(w), fe.grad(v))*fe.dx - (f - w*w)*v*fe.dx

# Solve
fe.solve(F == 0, w, bc)

#Plotting solutions and save it
c = fe.plot(w)
plt.colorbar(c)
plt.savefig('exam_nonlinear.pdf')

#Error on L2 space
error_L2 = fe.errornorm(w_D, w, 'L2')
print('error_L2  =', error_L2)
#
plt.show()
