from stable_baselines3 import PPO
from enviroment import CathRobotEnv
import mujoco.viewer
import time

# 1. Load the environment and the trained brain
env = CathRobotEnv()
model = PPO.load("drl_training/results/robot_weights.zip")
obs, info = env.reset()
print("Loaded the AI Brain! Opening Viewer...")

# 2. Open the 3D Viewer
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    
    # Set up the camera
    viewer.cam.distance = 5.5
    viewer.cam.elevation = -20
    viewer.cam.lookat[2] = 0.5

    while viewer.is_running():
        
        # 3. Ask the AI what to do based on what it sees!
        # deterministic=True tells the AI to use its best learned move (no random exploring)
        action, _states = model.predict(obs, deterministic=True)
        
        # 4. Take that action in the physics world
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 5. If the ball hits the floor (Terminated), reset and throw another one!
        if terminated or truncated:
            obs, info = env.reset()
            
        # 6. Update the graphics so our human eyes can see it
        viewer.sync()
        time.sleep(env.model.opt.timestep)