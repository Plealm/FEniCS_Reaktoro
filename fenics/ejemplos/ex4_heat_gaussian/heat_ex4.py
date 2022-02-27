"""Ecuación de calor con condiciones de Dirichlet con la función Gaussiana."""
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

Tomando el valor incial

       u0(x, y) = exp(-ax² -ay²)

en un dominion de [-2,2]x [-2, 2]
"""

T = 2.0             # Tiempo final
num_steps = 100      # numero de pasos
dt = T / num_steps  # diferencial de tiempo

# Creación de la malla
nx = ny = 10
mesh = fe.RectangleMesh(fe.Point(-2, -2), fe.Point(2, 2), nx, ny)
V = fe.FunctionSpace(mesh, 'P', 1)


def boundary(x, on_boundary):
    """Comprobación de si esta en la frontera."""
    return on_boundary


# expresión de la frontera de Dirichlet
bc = fe.DirichletBC(V, fe.Constant(0), boundary)

# Definición de las condicones de frontera
u_0 = fe.Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
                    degree=2, a=5)
# Definición del valor inicial
u_n = fe.interpolate(u_0, V)

# Define variational problem
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(0)

# Definición del problema
F = u*v*fe.dx + dt*fe.dot(fe.grad(u), fe.grad(v))*fe.dx - (u_n + dt*f)*v*fe.dx
a, L = fe.lhs(F), fe.rhs(F)
vtkfile = fe.File('heat_gaussian/solution.pvd')
# Definición de la solución
u = fe.Function(V)
t = 0
for n in range(num_steps):

    # Actulización de t
    t += dt

    # Solución de la ecuación diferencial
    fe.solve(a == L, u, bc)
    vtkfile << (u, t)
    # graficación
    d = fe.plot(u)

    # Error por vertices
    u_e = fe.interpolate(u_0, V)
    error = np.abs(u_e.vector() - u.vector()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

# Hold plot
plt.colorbar(d)
plt.show()
