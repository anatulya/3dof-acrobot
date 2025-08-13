import numpy as np
from math import cos, sin
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

"""
Plotting manipulator equations after euler-lagrange 
-> 1) derive lagrangian
-> 2) differentiate and plug partials + time derivative into euler-lagrange
-> 3) solve 2nd order diff. eq to find joint angles over time
-> 4) use joint angles to compute CoM of links over time 

For explanation of below equations, see 2dof_acrobot_derivation.pdf
"""

# mass, link lengths
m1, m2 = 1.0, 1.0
l1, l2 = 1.0, 1.0
g = 9.81

# actuate only 2nd joint
B = np.array([0.0, 1.0])

# sim
t_end = 10.0
dt = 0.005
t_eval = np.arange(0.0, t_end, dt)

# initial configuration (state space)
x0 = np.array([np.pi+0.1, 0.0, 0.0, 0.0])

def mat_inertial(q):
    """
        Computes M(q) [inertial matrix] from:
            joint angles q = [θ_1, θ_2]
    """
    th1, th2 = q

    m11 = (1/3)*m1*l1**2 + m2*l1**2 + m2*l1*l2*cos(th2) + (1/3)*m2*l2**2
    m12 = m21 = (1/2)*m2*l1*l2*cos(th2) + (1/3)*m2*l2**2
    m22 = (1/3)*m2*l2**2

    return np.array([
        [m11, m12],
        [m21, m22]
    ])

def mat_coriolis(q, qd):
    """
        Computes C(q,qdot) [coriolis matrix] from:
            joint angles q = [θ_1, θ_2]  
            joint angles time derivative qdot = [θdot_1, θdot_2]
    """

    th1, th2 = q
    d1, d2 = qd

    m11 = -m2*l1*l2*sin(th2)*d2
    m12 = -(1/2)*m2*l1*l2*sin(th2)*d2
    m21 = (1/2)*m2*l1*l2*d1*sin(th2)
    m22 = 0.0

    return np.array([
        [m11, m12],
        [m21, m22]
    ])

def vec_torque(q):
    """
    Computes torque vector from:
        joint angles q = [θ_1, θ_2] 
    """

    th1, th2 = q
    v1 = (1/2)*m1*g*l1*sin(th1) + m2*g*l1*sin(th1) + (1/2)*m2*g*l2*sin(th1+th2) 
    v2 = (1/2)*m2*g*l2*sin(th1+th2)

    return np.array([ v1, v2 ])

def f(t, x):
    th1, th2, d1, d2 = x
    q = np.array([ th1, th2 ])
    qd = np.array([ d1, d2 ])

    M = mat_inertial(q)
    C = mat_coriolis(q, qd)
    tau = vec_torque(q)

    # controller
    u = controller(x)

    # damping
    damping_coeff = 0.5
    tau_damping = damping_coeff * qd

    # M*qdd + C*qdot = Bu - tau
    qdd = np.linalg.solve(M,  B@u - tau - C@qd - tau_damping).flatten()

    return np.array([d1,d2,qdd[0],qdd[1]])

def fk(th1, th2):
    j1 = np.array([l1*sin(th1), -l1*cos(th1)])
    j2 = j1 + np.array([l2*sin(th1+th2), -l2*cos(th1+th2)])

    return np.array([
        [0.0, 0.0],
        j1,
        j2
    ])

sol = solve_ivp(f, (0, t_end), x0, t_eval=t_eval, rtol=1e-9, atol=1e-11, max_step=dt)

# animation
fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("x")
ax.set_ylabel("y")
line, = ax.plot([], [], marker='o', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(i):
    th1, th2 = sol.y[0, i], sol.y[1, i]
    p0, p1, p2 = fk(th1, th2)
    xs = [p0[0], p1[0], p2[0]]
    ys = [p0[1], p1[1], p2[1]]
    line.set_data(xs, ys)
    time_text.set_text(f"t={sol.t[i]:.2f}s")
    return line, time_text

ani = FuncAnimation(fig, update, frames=len(sol.t), init_func=init,
                    blit=True, interval=1000*dt, repeat=False)
plt.show()
