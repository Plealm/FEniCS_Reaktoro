"""Solución de la ecuación de Poisson."""
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import mshr as ms
"""
En dos dimensiones se puede expresar la ecuación de Poisson como:

        ∂² u     ∂² u
     -  ----- - ------ = f(x, y)
        ∂ x²     ∂ y²

Tal que, u es una función de dos variables definida sobre el espacio
bi-dimensional  Ω.

Para solucionar un problema PDE, es necesario seguir los siguientes pasos:

      1. Identificar las condiciones de frontera.
      2. Reformular la PDE como un problema variacional.
      3. Codificarlo en python.
      4. Visualizar los resultados.
"""

# Creación de la grilla
base = ms.Polygon([fe.Point(3,0),fe.Point(0,2.5),
                   fe.Point(-3, 0), fe.Point(0, -2.5)])
center = fe.Point(0.5,0.5)
horizontal_semi_axis = .5
vertical_semi_axis = .75
elipse = ms.Ellipse(center,
            horizontal_semi_axis,
            vertical_semi_axis)
circle1 = ms.Circle(fe.Point(1, 1), 0.25)
circle2 = ms.Circle(fe.Point(0, 1), 0.25)
domain = base - elipse - circle1 -circle2
mesh = ms.generate_mesh(domain, 64)
# definición de una función discreta sobre el espacio
V = fe.FunctionSpace(mesh, 'P', 1)

# Definición de las condiciones de frontera
u_D = fe.Expression('x[0]*x[0] +x[0]*x[1] + x[1]*x[1]', degree=2)


def boundary(x, on_boundary):
    """Comprobación de si esta en la frontera."""
    return on_boundary


# expresión de la frontera de Dirichlet
bc = fe.DirichletBC(V, u_D, boundary)

"""
Para la resolución por método variacional se multiplica por v y se integra,
talque:

       - ∫(∇² u) v dx = ∫fv dx

       - ∫(∇² u) v dx = ∫∇u⋅∇v dx - ∫(∂u/∂n)⋅v ds

El segundo término se anula dado que el la derivada de u de manera normal
a la fronte y como v debe ser 0 en la frontera donde u es conocido (v = 0).
Por lo tanto,

          ∫∇u⋅∇v dx = ∫fv dx

Es necesario discretizar el problema por lo que se usará u para la solución
discreta y uc para la solución continua

           a(u, v) = ∫∇u⋅∇v dx
           L(v) = ∫fv dx

           a(u, v) = L(v)

"""
# Definición del problema variacional
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(-6.0)
a = fe.dot(fe.grad(u), fe.grad(v))*fe.dx
L = f*v*fe.dx

# Se define la función u
u = fe.Function(V)
# Se soluciona la ecuación
fe.solve(a == L, u, bc)

# Grafica la solución
c = fe.plot(u)
plt.colorbar(c)
plt.savefig('dominio.pdf')
# Error de la norma al cuadrado
error_L2 = fe.errornorm(u_D, u, 'L2')

# El error dado por los vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
plt.show()
