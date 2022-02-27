"""Ecuación de calor con condiciones de Dirichlet."""

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

"""
Recordando la ecuación de calor es:
       ∂u
      ----   =  ∇²u + f
       ∂t

Es necesario discretizar la ecuación PDE
Por lo tanto,


       ∂u        u(n+1)  -  u(n)
      ----   =  -----------------
       ∂t                Δt

Entonces,

       u(n+1)  -  u(n)
       ----------------- = ∇²u + f,
               Δt

       u(n+1) - Δt∇²u = Δtf + u(n)

Sea

       a(u, v) = ∫(uv + Δt∇u⋅∇v) dx
       L(v) = ∫(u + Δtf)v dx

Se reescribe la ecuación como
       F(u, v) = a(u, v) - L(v) = 0
       F(u, v) = ∫(uv + Δt∇u⋅∇v) - (u + Δtf)v dx
"""

T = 2.0             # Tiempo final
num_steps = 20      # numero de pasos
dt = T / num_steps  # diferencial de tiempo
alpha = 3           # inicialización de alpha
beta = 1.2          # inicialización de beta

# Creación de la malla
nx = ny = 30
mesh = fe.UnitSquareMesh(nx, ny)
V = fe.FunctionSpace(mesh, 'P', 1)

# Definición de las condicones de frontera
u_D = fe.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                    degree=2, alpha=alpha, beta=beta, t=0)


def boundary(x, on_boundary):
    """Comprobación de si esta en la frontera."""
    return on_boundary


# expresión de la frontera de Dirichlet
bc = fe.DirichletBC(V, u_D, boundary)

# Definición del valor inicial
u_n = fe.interpolate(u_D, V)

# Define variational problem
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(beta - 2 - 2*alpha)

# Definición del problema
F = u*v*fe.dx + dt*fe.dot(fe.grad(u), fe.grad(v))*fe.dx - (u_n + dt*f)*v*fe.dx
a, L = fe.lhs(F), fe.rhs(F)

# Definición de la solución
u = fe.Function(V)
t = 0
for n in range(num_steps):

    # Actulización de t
    t += dt
    u_D.t = t

    # Solución de la ecuación diferencial
    fe.solve(a == L, u, bc)

    # graficación
    d = fe.plot(u)

    # Error por vertices
    u_e = fe.interpolate(u_D, V)
    error = np.abs(u_e.vector() - u.vector()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

# Hold plot
plt.colorbar(d)
plt.savefig('heat_solution.pdf')
plt.show()
