import mujoco
import mujoco.viewer

OPENARM_MODEL_PATH = "/home/hans/projects/openarm/openarm/simulation/models/openarm.xml"

model = mujoco.MjModel.from_xml_path(str(OPENARM_MODEL_PATH))
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
