"""Ecuación de Navier Stokes para un cilindro."""
import fenics as fe
import mshr as ms
import numpy as np
import matplotlib.pyplot as plt

T = 5.0             # Tiempo final
num_steps = 5000    # número de pasos
dt = T / num_steps  # diferencial temporal
mu = 0.001          # viscosidad dinamica
rho = 1             # densidad

# Creación de la malla
channel = ms.Rectangle(fe.Point(0, 0), fe.Point(2.2, 0.41))
cylinder = ms.Circle(fe.Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = ms.generate_mesh(domain, 64)

# Definición de las funciones del espacio
V = fe.VectorFunctionSpace(mesh, 'P', 2)
Q = fe.FunctionSpace(mesh, 'P', 1)

# Definición del contorno
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Defininición del perfil
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Definición de las condiciones de Diricleht
bcu_inflow = fe.DirichletBC(V, fe.Expression(inflow_profile, degree=2), inflow)
bcu_walls = fe.DirichletBC(V, fe.Constant((0, 0)), walls)
bcu_cylinder = fe.DirichletBC(V, fe.Constant((0, 0)), cylinder)
bcp_outflow = fe.DirichletBC(Q, fe.Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

# Definición funciones de prubea y de test
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
p = fe.TrialFunction(Q)
q = fe.TestFunction(Q)

# Definición de la evoluación de las soluciones
u_n = fe.Function(V)
u_  = fe.Function(V)
p_n = fe.Function(Q)
p_  = fe.Function(Q)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = fe.FacetNormal(mesh)
f  = fe.Constant((0, 0))
k  = fe.Constant(dt)
mu = fe.Constant(mu)
rho = fe.Constant(rho)


def epsilon(u):
    """Definición de la función tensión."""
    return fe.sym(fe.nabla_grad(u))


def sigma(u, p):
    """Definición de la función estres."""
    return 2*mu*epsilon(u) - p*fe.Identity(len(u))


# Definición del problema variacional para el primer paso
F1 = rho*fe.dot((u - u_n) / k, v)*fe.dx \
   + rho*fe.dot(fe.dot(u_n, fe.nabla_grad(u_n)), v)*fe.dx \
   + fe.inner(sigma(U, p_n), epsilon(v))*fe.dx \
   + fe.dot(p_n*n, v)*fe.ds - fe.dot(mu*fe.nabla_grad(U)*n, v)*fe.ds \
   - fe.dot(f, v)*fe.dx
a1 = fe.lhs(F1)
L1 = fe.rhs(F1)

# Definición del problema variacional para el segundo paso
a2 = fe.dot(fe.nabla_grad(p), fe.nabla_grad(q))*fe.dx
L2 = fe.dot(fe.nabla_grad(p_n), fe.nabla_grad(q))*fe.dx
- (1/k)*fe.div(u_)*q*fe.dx

# Definición del problema variacional para el tercer paso
a3 = fe.dot(u, v)*fe.dx
L3 = fe.dot(u_, v)*fe.dx - k*fe.dot(fe.nabla_grad(p_ - p_n), v)*fe.dx

# Union de matrices
A1 = fe.assemble(a1)
A2 = fe.assemble(a2)
A3 = fe.assemble(a3)

# Aplicación de las condiciones de contorno a las matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Creación de XDMF files
xdmffile_u = fe.XDMFFile('navier_stokes_cylinder/velocity.xdmf')
xdmffile_p = fe.XDMFFile('navier_stokes_cylinder/pressure.xdmf')

# Crecación de la serie temporal
timeseries_u = fe.TimeSeries('navier_stokes_cylinder/velocity_series')
timeseries_p = fe.TimeSeries('navier_stokes_cylinder/pressure_series')

# Guardar la malla
fe.File('navier_stokes_cylinder/cylinder.xml.gz') << mesh

# Creación de la barra de progreso
# progress = Progress('Time-stepping')
progress = fe.Progress('Time-stepping', num_steps)
# set_log_level(PROGRESS)

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Paso 1: Velocidad tentativa
    b1 = fe.assemble(L1)
    [bc.apply(b1) for bc in bcu]
    fe.solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Paso 2: Correciones de la presión
    b2 = fe.assemble(L2)
    [bc.apply(b2) for bc in bcp]
    fe.solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Paso 3: Correción a la velocidad
    b3 = fe.assemble(L3)
    fe.solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Guardar la solución en formato (XDMF/HDF5)
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    # Guardar valores nodales
    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    # Actualizar los anteriores valores
    u_n.assign(u_)
    p_n.assign(p_)

    # Update progress bar
    # progress.update(t / T)
    fe.set_log_level(fe.LogLevel.PROGRESS)
    progress += 1
    fe.set_log_level(fe.LogLevel.ERROR)
    print('u max:', u_.vector().get_local().max())

# Graficación
plt.figure()
fe.plot(u_, title='Velocity')
plt.savefig('velocidad.pdf')

plt.figure()
fe.plot(p_, title='Pressure')
plt.savefig('presión.pdf')
plt.show()
