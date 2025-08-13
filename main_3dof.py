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
x_star = np.array([math.pi, 0.0, 0.0, 0.0])
u_star = np.zeros(nu)
A, B = linearize(x_star, u_star)
print(f"[log] x: {x_star}, u: {u_star} controllable?: {is_controllable(A, B)}")
print(f"[log] eigen values of A: {np.linalg.eigvals(A)}")

Q = np.diag([10.0, 100.0, 1.0, 1.0])  # more position than velocity weight
R = 0.5*np.eye(nu)                      # was 0.05 → too “cheap” to use torque
K = lqr(A, B, Q, R)
print(f"[log] K shape: {K.shape}")


A_cl = A - B @ K
eig_cl = np.linalg.eigvals(A_cl)
print(f"[log] closed-loop eigvals: {eig_cl}")

def controller(x):
    q, qd = x[:nq], x[nv:]
    err_q = wrap_pi(q - x_star[:nq])
    err_qd = (qd - x_star[nv:])
    err = np.hstack([err_q, err_qd])

    # ---- static gravity feed-forward (qd = 0 to avoid Coriolis/damping) ----
    data.qpos[:] = q
    data.qvel[:] = 0.0               # gravity only; set =qd to cancel full bias if you prefer
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    # elbow DOF index (second hinge)
    elbow_dof = model.jnt_dofadr[1]  # qfrc_bias is in DOF space
    tau_g_elbow = data.qfrc_bias[elbow_dof]     # N·m at elbow
    u_ff = -tau_g_elbow                            # gear=1 → ctrl units

    # ---- LQR feedback ----
    u_fb = (u_star - K @ err).reshape(-1)         # (nu,)

    # combine FF + FB; if nu==1 we're elbow-only
    if nu == 1:
        u_cmd = np.array([u_ff]) + u_fb
    else:
        # generic: subtract gravity on each actuated DOF (approx)
        # NOTE: for multiple actuators you’d map joint torques → actuator space
        u_cmd = (-data.qfrc_bias[:nu]) + u_fb

    # debug
    # print("err_q:", err_q, "err_qd:", err_qd)
    # print("u_ff (grav):", u_ff, "u_fb:", u_fb, "u_cmd before sat:", u_cmd)

    return u_cmd

if __name__ == "__main__":
    # Bent start (radians) in the order your hinge joints are declared: [j0, j1, j2]
    data.qpos[:] = [math.pi, 0.2]
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