"""Solución de ecuación no linear de Poisson."""
import fenics as fe
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
import mshr as ms

"""
Se desea resolver la ecuación no linear de poisson:

               −∇⋅( q(u) ∇u )=f

El proceso variacional es:

              F(u;v) = ∫(q(u) ∇u⋅∇v − fv)dx

Con lo anterior se define
"""


def q(u):
    """Devuelve los valores de q(u)."""
    return -20*u + 5*u**2 + 60

# definición de los simbolos usados en FEniCS
x, y = sym.symbols('x[0], x[1]')
# Definición de la frontera
u = x + x*y - y**2
# Se usa Diff de sympy para recrear la ecuación
f = - sym.diff(q(u)*sym.diff(u, x), x) - sym.diff(q(u)*sym.diff(u, y), y)
f = sym.simplify(f)
# formato para escribir las expresiones de C/C++  a Fenics
u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)
print('u =', u_code)
print('f =', f_code)

center = fe.Point(0.,0.)
horizontal_semi_axis = 2
vertical_semi_axis = 2
elipse = ms.Ellipse(center,
            horizontal_semi_axis,
            vertical_semi_axis)
mesh = ms.generate_mesh(elipse, 64)
V = fe.FunctionSpace(mesh, 'P', 1)

u_D = fe.Expression(u_code, degree=4)

def boundary(x, on_boundary):
    return on_boundary

bc = fe.DirichletBC(V,u_D, boundary)

u = fe.Function(V)
v = fe.TestFunction(V)
f = fe.Expression(f_code, degree=4)
F = q(u)*fe.dot(fe.grad(u), fe.grad(v))*fe.dx - f*v*fe.dx

fe.solve(F == 0, u, bc)

c = fe.plot(u)
plt.colorbar(c)
plt.savefig('nonlinear_poisson.pdf')
plt.show()
