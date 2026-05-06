import mujoco
import mujoco.viewer
import time

# 1. Point to the specific robot model you want to load
# Let's start with the quadruped for the thesis!
model_path = "drl_training/models/scene.xml"

print(f"Loading model from: {model_path}")

# 2. Load the physics model and create a data state
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 3. Launch the interactive 3D viewer
print("Launching MuJoCo Viewer... (Press ESC to close)")
mujoco.viewer.launch(model, data)