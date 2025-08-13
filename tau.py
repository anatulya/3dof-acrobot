import numpy as np, mujoco
model = mujoco.MjModel.from_xml_path("model/test_rod_2dof.xml")
data  = mujoco.MjData(model)

def max_elbow_gravity_tau(model, data, grid=181, elbow_idx=1):
    qs = np.linspace(-np.pi, np.pi, grid)
    tau_max = 0.0
    for q0 in qs:
        for q1 in qs:
            data.qpos[:] = [q0, q1]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)     # qfrc_bias now = gravity (no coriolis at q̇=0)
            tau = abs(data.qfrc_bias[elbow_idx])
            if tau > tau_max:
                tau_max = tau
    return tau_max

tau_g = max_elbow_gravity_tau(model, data)   # N·m at elbow
ctrl_limit = 2.5 * tau_g                      # pick 2–3× headroom
print("Suggested |ctrl| limit:", ctrl_limit, "N·m")
