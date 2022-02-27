"""Ecuación de Navier-Stokes."""

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
"""
Recordando la ecuación de Navier-Stokes es:

            ( ∂u         )
         ϱ  (---- + u⋅∇u ) = ∇⋅σ(u, p) + f
            ( ∂t         )


            ∇⋅u = 0

Tal que, σ es el tensor de estres y f es la fuerza por unidad de volumen,
para un fluido newtoninano se tiene:

        σ(u, p) = 2μϵ(u)−pI

Donde ϵ(u) es el tensor de tensión.

        ε = 1/2 ( ∇u + (∇u)^T)
"""

T = 3.0             # Tiempo final
num_steps = 500     # número de pasos
dt = T / num_steps  # diferencial temporal
mu = 1              # viscosidad cinmetica
rho = 1             # densidad

# Creación de la malla y las funciones en el espacio
mesh = fe.UnitSquareMesh(16, 16)
V = fe.VectorFunctionSpace(mesh, 'P', 2)
Q = fe.FunctionSpace(mesh, 'P', 1)

# Definición de los limites
inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 1)'
walls = 'near(x[1], 0) || near(x[1], 1)'

# Definición de las condiciones por Dirichlet
bcu_noslip = fe.DirichletBC(V, fe.Constant((0, 0)), walls)
bcp_inflow = fe.DirichletBC(Q, fe.Constant(8), inflow)
bcp_outflow = fe.DirichletBC(Q, fe.Constant(0), outflow)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

# Definición de las funciones de prueba y de test
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
p = fe.TrialFunction(Q)
q = fe.TestFunction(Q)

# Definición de las funciones solución para diferentes tiempos
u_n = fe.Function(V)
u_ = fe.Function(V)
p_n = fe.Function(Q)
p_ = fe.Function(Q)

# Definición de las expresiones para el problema variacional
U = 0.5*(u_n + u)
n = fe.FacetNormal(mesh)
f = fe.Constant((0, 0))
k = fe.Constant(dt)
mu = fe.Constant(mu)
rho = fe.Constant(rho)


def epsilon(u):
    """Definición de la función tensión."""
    return fe.sym(fe.nabla_grad(u))


def sigma(u, p):
    """Definición de la función estres."""
    return 2*mu*epsilon(u) - p*fe.Identity(len(u))


# Definición del problema variacional a primer paso
F1 = rho*fe.dot((u - u_n) / k, v)*fe.dx + \
     rho*fe.dot(fe.dot(u_n, fe.nabla_grad(u_n)), v)*fe.dx \
     + fe.inner(sigma(U, p_n), epsilon(v))*fe.dx \
     + fe.dot(p_n*n, v)*fe.ds - fe.dot(mu*fe.nabla_grad(U)*n, v)*fe.ds \
     - fe.dot(f, v)*fe.dx
a1 = fe.lhs(F1)
L1 = fe.rhs(F1)

# Definición del problema variacional a segundo paso
a2 = fe.dot(fe.nabla_grad(p), fe.nabla_grad(q))*fe.dx
L2 = fe.dot(fe.nabla_grad(p_n),
            fe.nabla_grad(q))*fe.dx - (1/k)*fe.div(u_)*q*fe.dx

# Definición del problema variacional a tercer paso
a3 = fe.dot(u, v)*fe.dx
L3 = fe.dot(u_, v)*fe.dx - k*fe.dot(fe.nabla_grad(p_ - p_n), v)*fe.dx

# Union de las matrices
A1 = fe.assemble(a1)
A2 = fe.assemble(a2)
A3 = fe.assemble(a3)

# Aplicación de las condiciones de contorno a las matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# paso temporal
t = 0
for n in range(num_steps):

    # Actulización del tiempo
    t += dt

    # Paso 1: Paso de velocidad tentativo
    b1 = fe.assemble(L1)
    [bc.apply(b1) for bc in bcu]
    fe.solve(A1, u_.vector(), b1)

    # Paso 2: Paso de correción de la presión
    b2 = fe.assemble(L2)
    [bc.apply(b2) for bc in bcp]
    fe.solve(A2, p_.vector(), b2)

    # Step 3: la correción de la velocidad por paso
    b3 = fe.assemble(L3)
    fe.solve(A3, u_.vector(), b3)

    # Graficación
    fe.plot(u_)
    # Encontrar el error
    u_e = fe.Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = fe.interpolate(u_e, V)
    error = np.abs(u_e.vector() - u_.vector()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    print('max u:', u_.vector().max())

    # Guardar los resultados
    u_n.assign(u_)
    p_n.assign(p_)

plt.savefig('solución.pdf')
plt.show()
