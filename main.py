import mujoco
from mujoco import viewer
import math
import time

model = mujoco.MjModel.from_xml_path("model/test_rod.xml")
data  = mujoco.MjData(model)

# Bent start (radians) in the order your hinge joints are declared: [j0, j1, j2]
data.qpos[:] = [math.pi/4, 0.0, 0.0]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)  # recompute derived quantities

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)

        v.sync()
        time.sleep(0.001)