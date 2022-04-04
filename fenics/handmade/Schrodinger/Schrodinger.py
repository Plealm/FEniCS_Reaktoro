"""FEniCS implementation for Quantum Harmonic oscillator
and resolve using Crank-Nicolson method."""

from matplotlib import interactive
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

""" 
To use Crank-Nicolson method, we suppose u = p + iq
and the Schrodinger equation can be computed as

The real part :
      Δt
    ------  (∇² p^(n + 1) - Vp^(n + 1)) - q^(n + 1)
       2
                       Δt   
     = -q^(n + 1) +  ------ (Vp^(n) - ∇² p^(n))
                        2

And the Imaginary part:

      Δt
    ------  (∇² q^(n + 1) - Vq^(n + 1)) - p^(n + 1)
       2
                       Δt   
     = -p^(n + 1) +  ------ (Vq^(n) - ∇² q^(n))
                        2
"""

T = 2.0            # Final time
num_steps = 100    # Number of time steps
dt = T / num_steps # Time step size

# Definition of domain and kind of function
mesh = fe.RectangleMesh(fe.Point(-5, -5), fe.Point(5, 5), 55, 55)
V = fe.VectorFunctionSpace(mesh, 'Lagrange', 1, dim=2)

# Define boundary condition
u_0 = fe.Expression(('exp(-2*(pow(x[0], 2) + pow(x[1], 2)))', '0')
                 , degree=1)

# Define initial value
u_n = fe.interpolate(u_0, V)

def boundary(x, on_boundary):
    return on_boundary

# Dirichlet boundary
bc = fe.DirichletBC(V,(0, 0), boundary)

u = fe.Function(V)
v = fe.TestFunction(V)
# Potential
f = fe.Expression(('pow(x[0], 2) + pow(x[1], 2)'
                   , '0'), degree=2)



# Weak Form with u[0] = p and u[1] = q

ReVL = -u[1]*v[0]*fe.dx + (dt/2)*(-fe.dot(fe.grad(u[0]), fe.grad(v[0])) - 
        f[0]*u[0]*v[0])*fe.dx
ReHL = -u_n[1]*v[0]*fe.dx + (dt/2)*(f[0]*u_n[0]*v[0] + fe.dot(fe.grad(u_n[0]), 
        fe.grad(v[0])))*fe.dx
ImVL = u[0]*v[1]*fe.dx + (dt/2)*(-fe.dot(fe.grad(u[1]), fe.grad(v[1])) - 
        f[0]*u[1]*v[1])*fe.dx
ImHL = u_n[0]*v[1]*fe.dx + dt/2*(f[0]*u_n[1]*v[1] + fe.dot(fe.grad(u_n[1]), 
        fe.grad(v[1])))*fe.dx

# Equals zero 
FReal = ReVL - ReHL
FIm = ImVL - ImHL
# Combine all the equations together
Fny = FReal + FIm

t = 0

J = fe.derivative(Fny, u)
for n in range(num_steps):
        # Update current time
        t += dt
        # Compute solution
        fe.solve(Fny == 0, u, bc, J=J)
        # plot solution
        d = fe.plot(u)
        
        # Update previous solution
        u_n.assign(u)
# print the solution
plt.colorbar(d)
plt.savefig('wave_function_oscillator.pdf')
plt.show()