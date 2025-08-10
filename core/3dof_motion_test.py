import numpy as np
from math import pi
from sympy import symbols, Matrix, sin, cos
from sympy.physics.mechanics import (
    dynamicsymbols, ReferenceFrame, Point, RigidBody, inertia,
    KanesMethod, Particle
)
from sympy.utilities.lambdify import lambdify
from scipy.linalg import solve_continuous_are

# generalized coords/inputs
q1, q2, q3 = dynamicsymbols("q1 q2 q3")
u1, u2, u3 = dynamicsymbols("u1 u2 u3")

# parameters 
m1, m2, m3 = symbols('m1 m2 m3', positive=True)
l1, l2, l3 = symbols('l1 l2 l3', positive=True)
I1z, I2z, I3z = symbols('I1z I2z I3z', positive=True)  # planar inertias about z
g = symbols('g', positive=True)

# Actuation: set True where a joint is actuated.
# E.g., classic acrobot: [False, True] for 2-DOF; here maybe [False, True, True]
actuated = [False, True, True]

N = ReferenceFrame('N')