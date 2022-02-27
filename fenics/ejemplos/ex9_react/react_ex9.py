"""Solución de las ecuaciones de advección, difusión y reacción."""
import fenics as fe
import matplotlib.pyplot as plt

T = 5.0             # Tiempo final
num_steps = 500     # numero de pasos
dt = T / num_steps  # diferencial de tiempo
eps = 0.01          # coeficiente de difusión
K = 10.0            # rata de reacción

# Toma la malla anteriormente usada
mesh = fe.Mesh('../ex8_navier_cylinder/navier_stokes_cylinder/cylinder.xml.gz')

# Definición del espacio de la velocidad
W = fe.VectorFunctionSpace(mesh, 'P', 2)

# Defiición del espacio de las concentraciones
P1 = fe.FiniteElement('P', fe.triangle, 1)
element = fe.MixedElement([P1, P1, P1])
V = fe.FunctionSpace(mesh, element)

# Definición de las funciones test
v_1, v_2, v_3 = fe.TestFunctions(V)

# Definición de las funciones de concentración y velocidad
w = fe.Function(W)
u = fe.Function(V)
u_n = fe.Function(V)

# separación  de las funciones
u_1, u_2, u_3 = fe.split(u)
u_n1, u_n2, u_n3 = fe.split(u_n)

# Definición de los terminos base
f_1 = fe.Expression('pow(x[0]-0.1,2)+pow(x[1]-0.1,2)<0.05*0.05 ? 0.1 : 0',
                    degree=1)
f_2 = fe.Expression('pow(x[0]-0.1,2)+pow(x[1]-0.3,2)<0.05*0.05 ? 0.1 : 0',
                    degree=1)
f_3 = fe.Constant(0)

# Defininición de expresiones usadas en el formalismo variacional
k = fe.Constant(dt)
K = fe.Constant(K)
eps = fe.Constant(eps)

# Definición del problema variacional
F = ((u_1 - u_n1) / k)*v_1*fe.dx + fe.dot(w, fe.grad(u_1))*v_1*fe.dx \
  + eps*fe.dot(fe.grad(u_1), fe.grad(v_1))*fe.dx + K*u_1*u_2*v_1*fe.dx  \
  + ((u_2 - u_n2) / k)*v_2*fe.dx + fe.dot(w, fe.grad(u_2))*v_2*fe.dx \
  + eps*fe.dot(fe.grad(u_2), fe.grad(v_2))*fe.dx + K*u_1*u_2*v_2*fe.dx  \
  + ((u_3 - u_n3) / k)*v_3*fe.dx + fe.dot(w, fe.grad(u_3))*v_3*fe.dx \
  + eps*fe.dot(fe.grad(u_3), fe.grad(v_3))*fe.dx - K*u_1*u_2*v_3*fe.dx \
  + K*u_3*v_3*fe.dx \
  - f_1*v_1*fe.dx - f_2*v_2*fe.dx - f_3*v_3*fe.dx

# creacion de la serie temporal
timeseries_w = fe.TimeSeries('../ex8_navier_cylinder/' +
                             'navier_stokes_cylinder/velocity_series')

# Creación de archivos VTK
vtkfile_u_1 = fe.File('reaction_system/u_1.pvd')
vtkfile_u_2 = fe.File('reaction_system/u_2.pvd')
vtkfile_u_3 = fe.File('reaction_system/u_3.pvd')

# Creación de la barra de progreso
steps = num_steps
p = fe.Progress("Looping", steps)
# inicialización del tiempo
t = 0
for n in range(num_steps):

    # actualización del tiempo
    t += dt

    # lectura del archivo de la serie temporal
    timeseries_w.retrieve(w.vector(), t)

    # Solucion del problema variacional
    fe.solve(F == 0, u)

    # Guardar la solución
    _u_1, _u_2, _u_3 = u.split()
    vtkfile_u_1 << (_u_1, t)
    vtkfile_u_2 << (_u_2, t)
    vtkfile_u_3 << (_u_3, t)

    # Actualización de la solución anterior
    u_n.assign(u)

    # Actualización de la barra de progreso
    fe.set_log_level(fe.LogLevel.PROGRESS)
    p += 1
    fe.set_log_level(fe.LogLevel.ERROR)


plt.show()
