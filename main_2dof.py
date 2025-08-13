import math
import numpy as np
import scipy.linalg as la
import mujoco
from mujoco import viewer

# ========= Config =========
XML_PATH = "model/test_rod_2dof.xml"  # <- change this
USE_KEYFRAME = "init"   # set to None if you don't have <key name="init">
ANGLE_GATE_DEG = 15.0   # enable LQR when |q1 - pi| < this
SPEED_GATE = 2.0        # and ||qdot|| < this (rad/s)
EGAIN = 8.0             # energy-shaping gain (raise if it’s sluggish)
KD = 0.25               # damping on the actuated joint
V_CAP = 0.8             # (optional) Lyapunov gate — larger = earlier switch

# ========= Helpers =========
def wrap_pi(a): return (a + np.pi) % (2*np.pi) - np.pi
def statevec(q, v): return np.concatenate([q, v])
def split_state(x, nq): return x[:nq], x[nq:]

def maybe_apply_keyframe(model, data, keyname):
    if not keyname: return
    try:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyname)
        mujoco.mj_resetDataKeyframe(model, data, kid)
    except Exception:
        pass

def print_actuator_map(model):
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        an = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"actuator[{a}] '{an}' -> joint '{jn}'")

def ctrl_clip_fn(model):
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    has = np.isfinite(lo) & np.isfinite(hi) & (hi > lo + 1e-12)
    def clip(u):
        out = np.array(u, dtype=float, copy=True)
        out[has] = np.minimum(np.maximum(out[has], lo[has]), hi[has])
        return out
    return clip

# xdot via mj_forward (no integration)
def f_xdot(model, data, x, u):
    nq = model.nq
    q, v = split_state(x, nq)
    data.qpos[:] = q
    data.qvel[:] = v
    data.ctrl[:] = u
    mujoco.mj_forward(model, data)   # fills qacc
    return np.concatenate([v, data.qacc.copy()])

def linearize_fd(model, data, x_eq, u_eq, eps_x=1e-5, eps_u=1e-5):
    n, m = x_eq.size, u_eq.size
    A = np.zeros((n, n)); B = np.zeros((n, m))
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps_x
        fp = f_xdot(model, data, x_eq + dx, u_eq)
        fm = f_xdot(model, data, x_eq - dx, u_eq)
        A[:, i] = (fp - fm) / (2*eps_x)
    for j in range(m):
        du = np.zeros(m); du[j] = eps_u
        fp = f_xdot(model, data, x_eq, u_eq + du)
        fm = f_xdot(model, data, x_eq, u_eq - du)
        B[:, j] = (fp - fm) / (2*eps_u)
    return A, B

def lqr(A, B, Q, R):
    P = la.solve_continuous_are(A, B, Q, R)
    K = la.solve(R, B.T @ P)    # inv(R) @ B.T @ P
    return K, P

# ========= Load model =========
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
assert model.nq == 2 and model.nu == 1, "Expect 2 joints (j1,j2) and 1 actuator on j2."
maybe_apply_keyframe(model, data, USE_KEYFRAME)
mujoco.mj_forward(model, data)
print_actuator_map(model)
clip_u = ctrl_clip_fn(model)

# ========= Energies =========
def total_energy(model, data):
    return mujoco.mj_energyPos(model, data) + mujoco.mj_energyVel(model, data)

# target energy at upright (q1=pi, q2=0, qdot=0)
scratch = mujoco.MjData(model)
scratch.qpos[:] = 0.0; scratch.qvel[:] = 0.0
mujoco.mj_forward(model, scratch)
E_bottom = total_energy(model, scratch)

scratch.qpos[:] = 0.0; scratch.qvel[:] = 0.0
scratch.qpos[0] = math.pi
mujoco.mj_forward(model, scratch)
E_top = total_energy(model, scratch)
E_star = E_top

# ========= LQR (upright) =========
x_eq_top = statevec(np.array([math.pi, 0.0]), np.zeros(2))
u_eq = np.zeros(1)

A_t, B_t = linearize_fd(model, data, x_eq_top, u_eq)
Q_t = np.diag([300.0, 80.0, 6.0, 2.0])   # [q1,q2,q1d,q2d]
R_t = np.array([[0.05]])
K_t, P_t = lqr(A_t, B_t, Q_t, R_t)

# ========= Controllers =========
ANGLE_GATE = np.deg2rad(ANGLE_GATE_DEG)

def in_capture_region(q, v):
    # angle gate on main link, speed gate on total speed
    near = abs(wrap_pi(q[0] - math.pi)) < ANGLE_GATE
    slow = np.linalg.norm(v) < SPEED_GATE
    # optional Lyapunov gate for extra safety
    e = np.r_[wrap_pi(q - x_eq_top[:2]), v - x_eq_top[2:]]
    V = float(e @ (P_t @ e))
    return (near and slow and V < V_CAP), V

def lqr_top(q, v):
    e = np.r_[wrap_pi(q - x_eq_top[:2]), v - x_eq_top[2:]]
    return -K_t @ e   # shape (1,)

def energy_shaping(q, v):
    """
    u = EGAIN * (E* - E) * qdot2 - KD * qdot2
    (injects/removes energy via actuated joint velocity)
    """
    E = total_energy(model, data)
    qdot2 = v[1]
    u = EGAIN * (E_star - E) * qdot2 - KD * qdot2
    return np.array([u])

# ========= Main loop =========
mode = "swing"   # "swing" or "lqr"
last_switch = 0.0
DWELL_ON = 0.15
DWELL_OFF = 0.15

def key_callback(keycode):
    global mode, last_switch
    if keycode == ord('m'):
        mode = "lqr" if mode == "swing" else "swing"
        last_switch = data.time

with viewer.launch_passive(model, data, key_callback=key_callback) as v:
    tlog = 0.0
    while v.is_running():
        q = data.qpos.copy()
        vq = data.qvel.copy()

        can_enter, V = in_capture_region(q, vq)
        if mode == "swing":
            if (data.time - last_switch) > DWELL_OFF and can_enter:
                mode = "lqr"; last_switch = data.time
        else:  # lqr
            # if we fall far out, drop back to swing
            if (data.time - last_switch) > DWELL_ON and not can_enter:
                mode = "swing"; last_switch = data.time

        if mode == "swing":
            u = energy_shaping(q, vq)
        else:
            u = lqr_top(q, vq)

        data.ctrl[:] = clip_u(u)
        mujoco.mj_step(model, data)
        v.sync()

        if data.time - tlog > 0.25:
            print(f"t={data.time:5.2f}  mode={mode:5s}  |q1-π|={abs(wrap_pi(q[0]-math.pi))*180/np.pi:5.1f}°  "
                  f"||qd||={np.linalg.norm(vq):.2f}  V={V:.2f}")
            tlog = data.time
