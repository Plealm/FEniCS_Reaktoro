"""Solución de la ecuación de Poisson para un caso real."""
import fenics as fe
import mshr as ms
import numpy as np
import matplotlib.pyplot as plt

"""Deflección de una membrana."""

"""
Se soluciona la ecuación de poisson:

     - ∇² w = p

     p(x, y) = 4*exp(β²⋅x² + (y - R0)²)

Tal que p es una gausiana centrada en (0, 0.6).
Con condiciones de contorno


                w = 0  en el contorno
"""
# Creación de la malla
domain = ms.Circle(fe.Point(0, 0), 1)
mesh = ms.generate_mesh(domain, 64)
# Creación de la función en el espacio
V = fe.FunctionSpace(mesh, 'P', 2)

# Definición de las condiciones de frontera
w_D = fe.Constant(0)


def boundary(x, on_boundary):
    """Comprobación de si esta en la frontera."""
    return on_boundary


# expresión de la frontera de Dirichlet
bc = fe.DirichletBC(V, w_D, boundary)

# Definición de los parametros de la función de presión p(x,y)
beta = 8
R0 = 0.6
p = fe.Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))',
                  degree=1, beta=beta, R0=R0)

# Problema variacional
w = fe.TrialFunction(V)
v = fe.TestFunction(V)
a = fe.dot(fe.grad(w), fe.grad(v))*fe.dx
L = p*v*fe.dx

# Solución
w = fe.Function(V)
fe.solve(a == L, w, bc)

""" Here we examine the membrane's response to the pressure,
    so it must transform this function into a finite element function"""
p = fe.interpolate(p, V)
plt.figure()
D = fe.plot(w, title='Deflection')
plt.colorbar(D)
plt.figure()
Lo = fe.plot(p, title='Load')
plt.colorbar(Lo)

# Guardar la solución en un archivo formato vtk
vtkfile_w = fe.File('poisson_membrane/deflection.pvd')
vtkfile_w << w
vtkfile_p = fe.File('poisson_membrane/load.pvd')
vtkfile_p << p
# Curve plot along x = 0 comparing p and w
tol = 0.001  # avoid hitting points outside the domain
y = np.linspace(-1 + tol, 1 - tol, 101)
points = [(0, y_) for y_ in y]  # 2D points
w_line = np.array([w(point) for point in points])
p_line = np.array([p(point) for point in points])
plt.figure()
plt.plot(y, 50*w_line, 'k', linewidth=2)  # magnify w
plt.plot(y, p_line, 'b--', linewidth=2)
plt.grid(True)
plt.xlabel('$y$')
plt.legend(['Deflection ($\\times 50$)', 'Load'], loc='upper left')
plt.savefig('poisson_membrane/curves.pdf')
plt.savefig('poisson_membrane/curves.png')

# Hold plots
plt.show()
