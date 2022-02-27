"""Solución de la ecuación de no lineal de Poisson."""
import fenics as fe
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np

"""
La ecuación diferencial no lineal de Poisson:

              -∇⋅(q(u)∇u) = f,

Para el metodo variacional se toma

             F = ∫(q(u)∇u⋅∇v - fv)dx
"""


def q(u):
    """Retorna un coeficiente no lineal."""
    return 1 + u**2


# uso de Sympy para escribir la solución de u respecto a f
x, y = sym.symbols('x[0], x[1]')
u = 1 + x + 2*y
f = -sym.diff(q(u)*sym.diff(u, x), x) - sym.diff(q(u)*sym.diff(u, y), y)
f = sym.simplify(f)
u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)
print('u = ', u_code)
print('f = ', f_code)

# Creación de la malla
mesh = fe.UnitSquareMesh(80, 80)
V = fe.FunctionSpace(mesh, 'P', 1)

# Definición de las condicones de frontera
u_D = fe.Expression(u_code, degree=2)


def boundary(x, on_boundary):
    """Comprobación de si esta en la frontera."""
    return on_boundary


# expresión de la frontera de Dirichlet
bc = fe.DirichletBC(V, u_D, boundary)

# Problema variacional
u = fe.Function(V)  # Nota: NO TrialFunction!
v = fe.TestFunction(V)
f = fe.Expression(f_code, degree=2)
F = q(u)*fe.dot(fe.grad(u), fe.grad(v))*fe.dx - f*v*fe.dx

# Solución
fe.solve(F == 0, u, bc)

# gráficación
c = fe.plot(u)
plt.colorbar(c)
# guardar la solución
plt.savefig('nonlinear_poisson.pdf')
# Error por vertices
u_e = fe.interpolate(u_D, V)
error = np.abs(u_e.vector() - u.vector()).max()
print(f'error_max = {error:.5f}')
plt.show()
