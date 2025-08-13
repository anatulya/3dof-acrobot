import math, time
import numpy as np
import scipy.linalg as la
import mujoco
from mujoco import viewer

# ------------- Config -------------
XML_PATH     = "model/test_rod.xml"   # your 3-DOF acrobot MJCF
TARGET       = "top"                  # "top" swing-up & catch, or "bottom" stabilize downward
INIT_KEY     = "init"                 # if MJCF has <keyframe name="init">
PRINT_LINSYS = True

# Capture region gates (tune a bit per model)
ANGLE_CAP_DEG = 12.0
SPEED_CAP     = 1.5
V_CAP         = 0.6
V_STAY        = 0.35
DWELL_ON_S    = 0.15
DWELL_OFF_S   = 0.15

# Swing-up heuristic (now for j2 and j3)
PUMP_K2       = 3.0      # Nm on j2
PUMP_K3       = 4.5      # Nm on j3 (usually a bit stronger)
PUMP_DAMP2    = 0.20     # viscous damping on j2
PUMP_DAMP3    = 0.25     # viscous damping on j3

# ------------- Utilities -------------
def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def split_state(x, nq):
    return x[:nq], x[nq:]

def statevec(q, v):
    return np.concatenate([q, v])

def f_xdot(model, data, x, u):
    nq = model.nq
    q, v = split_state(x, nq)
    data.qpos[:] = q
    data.qvel[:] = v
    data.ctrl[:] = u
    mujoco.mj_forward(model, data)  # fills qacc
    return np.concatenate([v, data.qacc.copy()])

def linearize_fd(model, data, x_eq, u_eq, eps_x=1e-5, eps_u=1e-5):
    n = x_eq.size
    m = u_eq.size
    A = np.zeros((n, n))
    B = np.zeros((n, m))
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
    K = la.solve(R, B.T @ P)
    return K, P

def angle_error(q, q_ref):
    return wrap_pi(q - q_ref)

def ctrl_clip_fn(model):
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    has = np.isfinite(lo) & np.isfinite(hi) & (hi > lo + 1e-12)
    def clip(u):
        u = np.asarray(u).copy()
        if u.ndim == 0: u = np.array([u])
        out = u.copy()
        out[has] = np.minimum(np.maximum(out[has], lo[has]), hi[has])
        return out
    return clip

def maybe_apply_keyframe(model, data, key_name):
    try:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        mujoco.mj_resetDataKeyframe(model, data, kid)
    except Exception:
        pass

def print_actuator_map(model):
    # Helpful sanity check: which actuator maps to which joint?
    for a in range(model.nu):
        j_id = model.actuator_trnid[a, 0]
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) if j_id >= 0 else "none"
        a_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        print(f"actuator[{a}] '{a_name}' -> joint '{j_name}'")

# ------------- Load model & basic shapes -------------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
nq, nv, nu = model.nq, model.nv, model.nu
assert nq == nv, "This script assumes hinge-only joints so nq == nv."
maybe_apply_keyframe(model, data, INIT_KEY)
mujoco.mj_forward(model, data)

print_actuator_map(model)  # you should see u2->j2 and u3->j3 (order = XML order)

clip_u = ctrl_clip_fn(model)

# ------------- Define equilibria -------------
x_eq_bottom = statevec(np.zeros(nq), np.zeros(nv))
q_top = np.zeros(nq); q_top[0] = math.pi
x_eq_top = statevec(q_top, np.zeros(nv))
u_eq = np.zeros(nu)

# ------------- Linearize & build LQRs -------------
print("Linearizing at bottom...")
A_b, B_b = linearize_fd(model, data, x_eq_bottom, u_eq)
Q_b = np.diag([100.0]*nq + [2.0]*nv)
R_b = 0.05*np.eye(nu)
K_b, P_b = lqr(A_b, B_b, Q_b, R_b)

print("Linearizing at top...")
A_t, B_t = linearize_fd(model, data, x_eq_top, u_eq)
Q_t = np.diag([300.0]*nq + [5.0]*nv)
R_t = 0.05*np.eye(nu)
K_t, P_t = lqr(A_t, B_t, Q_t, R_t)

if PRINT_LINSYS:
    print(f"||A_b||={np.linalg.norm(A_b):.2e}, ||B_b||={np.linalg.norm(B_b):.2e}")
    print(f"||A_t||={np.linalg.norm(A_t):.2e}, ||B_t||={np.linalg.norm(B_t):.2e}")

# ------------- Capture tests & modes -------------
ANGLE_CAP = np.deg2rad(ANGLE_CAP_DEG)
def V_quad(e, P): return float(e @ (P @ e))

def in_capture_region(q, v):
    if TARGET == "bottom":
        q_err = angle_error(q, x_eq_bottom[:nq]); v_err = v - x_eq_bottom[nq:]
        e = np.r_[q_err, v_err]; V = V_quad(e, P_b)
    else:
        q_err = angle_error(q, x_eq_top[:nq]);    v_err = v - x_eq_top[nq:]
        e = np.r_[q_err, v_err]; V = V_quad(e, P_t)
    near_angle = abs(q_err[0]) < ANGLE_CAP
    slow = np.linalg.norm(v) < SPEED_CAP
    return near_angle and slow and (V < V_CAP), V

def lqr_control(q, v):
    if TARGET == "bottom":
        q_err = angle_error(q, x_eq_bottom[:nq]); v_err = v - x_eq_bottom[nq:]
        e = np.r_[q_err, v_err]; u = -K_b @ e
    else:
        q_err = angle_error(q, x_eq_top[:nq]);    v_err = v - x_eq_top[nq:]
        e = np.r_[q_err, v_err]; u = -K_t @ e
    return u

def swingup_torque(q, v):
    """
    Simple dual-actuator 'pump':
    - Use link-1 angle as phase reference.
    - Apply bang-bang torques with light joint-velocity damping on j2 & j3.
    Assumes actuators in XML map to j2 then j3 (check printout).
    """
    phase = wrap_pi(q[0])
    # sign(sin(phase)) switches direction each half-swing
    bang = np.sign(np.sin(phase))
    u = np.zeros(nu)
    if nu >= 1:
        u[0] = PUMP_K2 * bang - PUMP_DAMP2 * v[1]  # actuator 0 -> j2
    if nu >= 2:
        u[1] = PUMP_K3 * bang - PUMP_DAMP3 * v[2]  # actuator 1 -> j3
    return u

# ------------- Main sim loop -------------
paused = False
mode = "swing"      # "swing" or "lqr"
last_switch_t = 0.0

def key_callback(keycode):
    global paused, mode, last_switch_t
    if keycode == ord(' '):   # pause
        paused = not paused
    elif keycode == ord('m'): # toggle mode manually
        mode = "lqr" if mode == "swing" else "swing"
        last_switch_t = data.time

with viewer.launch_passive(model, data, key_callback=key_callback) as v:
    while v.is_running():
        if not paused:
            q = data.qpos.copy()
            vq = data.qvel.copy()

            # Mode logic w/ hysteresis
            can_enter, V_now = in_capture_region(q, vq)
            if mode == "swing":
                if (data.time - last_switch_t) > DWELL_OFF_S and can_enter:
                    mode = "lqr"; last_switch_t = data.time
            else:  # mode == "lqr"
                if TARGET == "bottom":
                    q_err = angle_error(q, x_eq_bottom[:nq]); v_err = vq - x_eq_bottom[nq:]
                    e = np.r_[q_err, v_err]; V_now = V_quad(e, P_b)
                else:
                    q_err = angle_error(q, x_eq_top[:nq]);    v_err = vq - x_eq_top[nq:]
                    e = np.r_[q_err, v_err]; V_now = V_quad(e, P_t)
                if (data.time - last_switch_t) > DWELL_ON_S and V_now > (V_STAY * 3.0):
                    mode = "swing"; last_switch_t = data.time

            # Control
            if mode == "swing":
                u = swingup_torque(q, vq)
            else:
                u = lqr_control(q, vq)

            # Clip to actuator limits and apply
            data.ctrl[:] = ctrl_clip_fn(model)(u)

            mujoco.mj_step(model, data)

        v.sync()
        time.sleep(0.001)
