"""Ecuación de elasticidad lineal."""

import fenics as fe
import matplotlib.pyplot as plt
from ufl import nabla_div

"""
Recordando la ecuación de elasticidad es:

           -∇⋅σ = f

       σ = λtr(ε)I + 2με
       ε = 1/2 ( ∇u + (∇u)^T)
Tal que, σ es el tensor de estres, f es la fuerza por unidad de volumen,
λ y μ son los parametros de Lamde.


La formulación variacional es:

        -∫(∇⋅σ)⋅v dx = ∫f⋅v dx
        -∫(∇⋅σ)⋅v dx =  ∫σ:∇v dx - ∫(σ⋅n)⋅v ds

donde el operador : es el producto interno entre tensores. Tomando σ⋅n = T
y sustituyendo

        ∫σ:∇v dx =  ∫f⋅v dx + ∫T⋅v ds

Entonces,

        a(u,v) = ∫σ:∇v dx = ∫σ:ε dx
        σ(u) = λ(∇⋅u)I + μ( ∇u + (∇u^T))
        L(v) = ∫f⋅v dx + ∫T⋅v ds
"""

# Definición de variables
L = 1
W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

# creación de la malla
mesh = fe.BoxMesh(fe.Point(0, 0, 0), fe.Point(L, W, W), 10, 3, 3)
V = fe.VectorFunctionSpace(mesh, 'P', 1)

# Condición de frontera
tol = 1E-14


def clamped_boundary(x, on_boundary):
    """Comprobación de si esta en la frontera."""
    return on_boundary and x[0] < tol


# expresión de la frontera de Dirichlet
bc = fe.DirichletBC(V, fe.Constant((0, 0, 0)), clamped_boundary)


def epsilon(u):
    """Definición de la función tensión."""
    return 0.5*(fe.nabla_grad(u) + fe.nabla_grad(u).T)


def sigma(u):
    """Definición de la función estres."""
    return lambda_*nabla_div(u)*fe.Identity(d) + 2*mu*epsilon(u)


# Definición del problema variacional
u = fe.TrialFunction(V)
d = u.geometric_dimension()  # Dimensión espacial
v = fe.TestFunction(V)
f = fe.Constant((0, 0, -rho*g))
T = fe.Constant((0, 0, 0))
a = fe.inner(sigma(u), epsilon(v))*fe.dx
L = fe.dot(f, v)*fe.dx + fe.dot(T, v)*fe.ds

# Solución
u = fe.Function(V)
fe.solve(a == L, u, bc)

# graficación solución
plt.figure()
plt.clf()
fe.plot(u, title='Displacement', mode='displacement')
vtkfile_u = fe.File('elasticity/solution.pvd')
vtkfile_u << u

plt.savefig('u.png')

# graficación estres
s = sigma(u) - (1./3)*fe.tr(sigma(u))*fe.Identity(d)  # deviatoric stress
von_Mises = fe.sqrt(3./2*fe.inner(s, s))
V = fe.FunctionSpace(mesh, 'P', 1)
von_Mises = fe.project(von_Mises, V)
plt.figure()
plt.clf()
fe.plot(von_Mises, title='Stress intensity')
vtkfile_von = fe.File('elasticity/stress.pvd')
vtkfile_von << von_Mises
plt.savefig('stress.png')

# Compute magnitude of displacement
u_magnitude = fe.sqrt(fe.dot(u, u))
u_magnitude = fe.project(u_magnitude, V)
plt.figure()
plt.clf()
fe.plot(u_magnitude, 'Displacement magnitude')
vtkfile_m = fe.File('elasticity/Displacement.pvd')
vtkfile_m << u_magnitude
plt.savefig('u_magnitude.png')
print('min/max u:',
      u_magnitude.vector().min(),
      u_magnitude.vector().max())

plt.show()
