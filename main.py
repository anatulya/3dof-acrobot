import time, numpy as np, mujoco, mujoco.viewer

model = mujoco.MjModel.from_xml_path("model/3dof_acrobot.xml")
data  = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # "start"

# Hold a joint target (j0 passive)
qref = data.qpos.copy()
qref[:3] = [0, 1, -1]  # j1, j2 targets
Kp = np.array([0.0, 40.0, 35.0])
Kd = np.array([0.0,  6.0,  5.0])

with mujoco.viewer.launch_passive(model, data) as v:
    t0 = time.time()
    while v.is_running():
        


        mujoco.mj_step(model, data); v.sync()
