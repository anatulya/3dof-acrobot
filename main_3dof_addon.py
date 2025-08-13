import mujoco
from mujoco import viewer
import math
import time
import numpy as np
from scipy import linalg

"""
Process of performing LQR:
    1) get xdot, linearize at a fixed point and obtain A, B matrices
        - how to linearize? -> perform finite differences to approximate jacobians of f(x,u) at x* and u*, wrt. x or u
        - central diff. formula: (f+ - f-) / 2ε, where f+ = f(x+ε, u), f- = f(x-ε, u), and same for u as well
    2) use a,b,q,r to perform CARE (continuous algebraic ricatti equation) and receive K
    3) use u(t) = -K*x(t) to get the control inputs <-- if K exists
"""

# mujoco
model = mujoco.MjModel.from_xml_path("model/test_roddy.xml")
data  = mujoco.MjData(model)
nq, nv, nu = model.nq, model.nv, model.nu
ctrl_limit = ( model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1] )

print(f"[log] nq: {nq}, nv: {nv}, nu: {nu}")

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def saturate(u):
    return np.clip(u, -10, 10)

def f_xu(x, u):
    """
    Compute xdot given x,u (xdot = [ qd, qdd ])
    Calls mj_forward to obtain qdd
    """

    data.qpos[:] = x[:nq]
    data.qvel[:] = x[nv:]
    data.ctrl[:] = u

    mujoco.mj_forward(model,data)

    return np.concatenate([x[nv:], data.qacc.copy()])

def linearize(x_star, u_star, eps_x=1e-6, eps_u=1e-6):
    """
    Finite differences linearization (central)
    Output: jacobian of f_xu wrt. x (A), and u (B) @ x*, u*
    """
    n, m = x_star.size, u_star.size

    A = np.zeros(( n, n ))
    B = np.zeros(( n, m ))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps_x

        f_plus = f_xu(x_star+dx, u_star)
        f_minus = f_xu(x_star-dx, u_star)
        A[:, i] = (f_plus - f_minus) / (2*eps_x)

    for i in range(m):
        du = np.zeros(m)
        du[i] = eps_u

        f_plus = f_xu(x_star, u_star + du)
        f_minus = f_xu(x_star, u_star - du)
        B[:, i] = (f_plus - f_minus) / (2*eps_u)

    return A, B

def is_controllable(A, B):
    n = A.shape[0]
    ctrb = B
    for i in range(1, n):
        ctrb = np.hstack((ctrb, np.linalg.matrix_power(A, i) @ B))
    rank = np.linalg.matrix_rank(ctrb)
    return rank == n



def lqr(A, B, Q, R):
    print(f"[log] A shape: {A.shape}, B shape: {B.shape}, Q shape: {Q.shape}, R shape: {R.shape}")

    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    return K

# lqr 
x_star = np.array([math.pi, 0.06, 0.0, 0.0])
u_star = np.zeros(nu)
A, B = linearize(x_star, u_star)
print(f"[log] x: {x_star}, u: {u_star} controllable?: {is_controllable(A, B)}")
print(f"[log] eigen values of A: {np.linalg.eigvals(A)}")

Q = np.diag([10.0, 10.0, 1.0, 1.0])  # more position than velocity weight
R = np.eye(nu)                      # was 0.05 → too “cheap” to use torque
K = lqr(A, B, Q, R)
print(f"[log] K shape: {K.shape}")


A_cl = A - B @ K
eig_cl = np.linalg.eigvals(A_cl)
print(f"[log] closed-loop eigvals: {eig_cl}")

# globals
UMAX = float(model.actuator_ctrlrange[:,1].max() if model.actuator_ctrlrange.size else 8.0)
alpha = 0.2                 # control smoothing
u_prev = np.zeros(nu)
deadband_q1 = 0.008         # ~0.5°
deadband_qd1 = 0.2          # rad/s
kd_elbow = 0.1              # extra viscous damping injection

def controller(x, dt=0.002):
    q  = x[:nq]
    qd = x[nq:]                     # (fix: use nq, not nv)

    # error (wrap ONLY angles)
    e_q  = wrap_pi(q - x_star[:nq])
    e_qd =       (qd - x_star[nq:])
    e = np.hstack([e_q, e_qd])

    # ---- full bias feed-forward (gravity + coriolis + passive) ----
    data.qpos[:] = q
    data.qvel[:] = qd               # << use current qd to cancel full bias
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    elbow_dof = model.jnt_dofadr[1]
    u_ff = np.array([-data.qfrc_bias[elbow_dof]])  # MuJoCo: u_ff = -qfrc_bias

    # ---- LQR feedback ----
    e_for_fb = e.copy()

    # deadband on elbow correction (prevents hunting at q1≈0)
    if abs(e_q[1]) < deadband_q1 and abs(e_qd[1]) < deadband_qd1:
        e_for_fb[1] = 0.0   # ignore elbow angle error
        e_for_fb[3] = 0.0   # ignore elbow velocity error

    u_fb = -(K @ e_for_fb).reshape(-1)

    # small elbow viscous damping injection
    u_damp = np.array([-kd_elbow * qd[1]])

    u_cmd = u_ff + u_fb + u_damp

    # ---- soft saturation + mild rate limiting + smoothing ----
    u_soft = UMAX * np.tanh(u_cmd / (0.9*UMAX))
    du_max = 100.0 * dt             # N·m per step (tune)
    u_step = np.clip(u_soft - u_prev, -du_max, du_max)
    u = alpha*u_prev + (1 - alpha) * (u_prev + u_step)
    u_prev[:] = u
    return u


if __name__ == "__main__":
    # Bent start (radians) in the order your hinge joints are declared: [j0, j1, j2]
    data.qpos[:] = [math.pi+0.3, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  # recompute derived quantities

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            x = np.concatenate(( data.qpos, data.qvel ))
            u = controller(x)
            print(f"[log] torque: {u}")
            data.ctrl[:] = u

            mujoco.mj_step(model, data)

            v.sync()
            time.sleep(0.002)