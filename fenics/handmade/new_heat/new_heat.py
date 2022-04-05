"""
Here we gonna solve a no linear poisson problem with a temporal derivate, i.e.,

    ∂_t u - ∇²u +u² = f

This is like heat equation plus the no linear term u², so the method to resolve it,
it's similar to already implemented method of heat equation
"""
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import mshr as ms
import sympy as sym

T = 2.0             # Tiempo final
num_steps = 20      # numero de pasos
dt = T / num_steps  # diferencial de tiempo
alpha = 3           # inicialización de alpha
beta = 1.2          # inicialización de beta

"""x, y, t = sym.symbols('x[0], x[1], x[2]')
u = 1 + x*x + alpha*y*y + beta*t

f = sym.diff(u, t) - sym.diff(sym.diff(u, x), x) \
    - sym.diff(sym.diff(u, y), y) + u**2
f = sym.simplify(f)

u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)
print('u =', u_code)
print('f =', f_code)"""

nx = ny = 30
mesh = fe.UnitSquareMesh(nx, ny)
V = fe.FunctionSpace(mesh, 'P', 1)


#u_D = fe.Expression(w_code, degree=2)
u_D = fe.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                    degree=2, alpha=alpha, beta=beta, t=0)

#Check boundary points
def boundary(x, on_boundary):
    return on_boundary

bc = fe.DirichletBC(V, u_D, boundary)

u_n = fe.interpolate(u_D, V)

u = fe.TrialFunction(V) #not TrialFunction
#u = fe.Function(V)
v = fe.TestFunction(V)
#f = fe.Expression('beta - 2 - 2*alpha + (1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t)**2',
#                  degree = 4, alpha=alpha, beta=beta, t=0)
f = fe.Constant(beta - 2 - 2*alpha)


# Definición del problema
F = u*v*fe.dx + dt*fe.dot(fe.grad(u), fe.grad(v))*fe.dx \
    + dt*u*u*v*fe.dx - (u_n + dt*f)*v*fe.dx
a, L = fe.lhs(F), fe.rhs(F)

# Definición de la solución
u = fe.Function(V)
t = 0
for n in range(num_steps):

    # Actulización de t
    t += dt
    u_D.t = t
    #f.t = t

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
plt.savefig('new_heat_solution.pdf')
plt.show()
