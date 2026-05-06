import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import mujoco.viewer


class CathRobotEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.model_path = "drl_training/models/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (7,),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (20,),
            dtype = np.float32
        )

    def _get_obs(self):
        # 0 to 6 on the Position list
        joint_angles = self.data.qpos[:7]
        # 0 to 6 on the Speed list
        joint_speeds = self.data.qvel[:7] 

        ball_position = self.data.body("target_object").xpos
        ball_velocity = self.data.body("target_object").cvel[3:]

        obs = np.concatenate([
            joint_angles, 
            joint_speeds, 
            ball_position, 
            ball_velocity
        ]).astype(np.float32)

        return obs


    def _get_info(self):
        
        return {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:7] = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0])
        self.data.qvel[:7] = np.zeros(7)
        
        distance = self.np_random.uniform(1.2, 1.8)
        angle = self.np_random.uniform(-np.pi/6, np.pi/6 )
        height_z = self.np_random.uniform(0.2, 0.6)
        spawn_x, spawn_y = -distance * np.cos(angle), distance * np.sin(angle)

        self.data.qpos[12:15] = [spawn_x, spawn_y, height_z]

        speed = self.np_random.uniform(2.5, 3.5)
        noise = self.np_random.uniform(-1.0, 1.0)

        vx = speed * np.cos(angle)
        vy = (-speed * np.sin(angle)) + noise  

        vz = self.np_random.uniform(1.5, 2.5)  
        self.data.qvel[12:15] = [vx, vy, vz] 

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), self._get_info()

    def step(self):
        
        
if __name__ == "__main__":
    # 1. Initialize our environment and trigger the first random throw
    env = CathRobotEnv()
    obs, info = env.reset()
    print("Environment successfully initialized! Opening 3D Viewer...")
    print(obs)

    # 2. Open the MuJoCo Interactive Viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        
        # --- ADD THESE 3 LINES ---
        viewer.cam.distance = 5.5      # Zoom out 3.5 meters
        viewer.cam.elevation = -20     # Look down at a 20-degree angle
        viewer.cam.lookat[2] = 0.5     # Focus on the robot's chest height
        # -------------------------
        while viewer.is_running():
            
            # Step the raw physics forward by 1 millisecond
            mujoco.mj_step(env.model, env.data)
            
            # Sync the graphics to match the math
            viewer.sync()
            
            # Slow down the loop so it looks like real-time to human eyes
            # (Otherwise the ball drops in 0.001 seconds!)
            time.sleep(env.model.opt.timestep)

