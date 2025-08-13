import time, math
import numpy as np
import mujoco
from mujoco import viewer
from scipy import linalg

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
XML_PATH = "model/test_rod_2dof.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
nq, nv, nu = model.nq, model.nv, model.nu
assert nq == nv == 2, f"Expect 2-DOF; got nq={nq}, nv={nv}"
assert nu >= 1, "Need at least the elbow actuator"

print(f"[log] nq={nq}, nv={nv}, nu={nu}, dt={model.opt.timestep}")
CTRL_LO = model.actuator_ctrlrange[:,0] if model.actuator_ctrlrange.size else np.array([-np.inf]*nu)
CTRL_HI = model.actuator_ctrlrange[:,1] if model.actuator_ctrlrange.size else np.array([ np.inf]*nu)
U_MAX = float(CTRL_HI.max())

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def wrap_pi(a):  # elementwise wrap to [-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

def soft_clip(u, umax=U_MAX):
    # smooth roll-off to avoid bang-bang near limits
    return umax * np.tanh(u / (0.9 * umax))

def f_xu(x, u):
    """xdot = [qd, qdd] at state x and input u (MuJoCo forward)."""
    data.qpos[:] = x[:nq]
    data.qvel[:] = x[nq:]
    data.ctrl[:] = u
    mujoco.mj_forward(model, data)
    return np.hstack([x[nq:], data.qacc.copy()])

def linearize(x_star, u_star):
    """Central-difference A = df/dx, B = df/du with well-scaled steps."""
    n, m = x_star.size, u_star.size
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    # Component-wise FD steps (critical for stable Jacobians)
    eps_x = np.array([1e-4, 1e-4, 1e-3, 1e-3])  # [q1,q2,qd1,qd2]
    u_max = U_MAX if np.isfinite(U_MAX) else 10.0
    eps_u = np.full(m, max(1e-3, 0.02 * u_max))

    for i in range(n):
        dx = np.zeros(n); dx[i] = eps_x[i]
        A[:, i] = (f_xu(x_star + dx, u_star) - f_xu(x_star - dx, u_star)) / (2 * eps_x[i])

    for j in range(m):
        du = np.zeros(m); du[j] = eps_u[j]
        B[:, j] = (f_xu(x_star, u_star + du) - f_xu(x_star, u_star - du)) / (2 * eps_u[j])

    return A, B

def is_controllable(A, B):
    n = A.shape[0]
    ctrb = B
    for i in range(1, n):
        ctrb =
