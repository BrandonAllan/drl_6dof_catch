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
        self.step_count = 0
        self.joint_mins = np.array([-6.28, -6.28, -3.14, -6.28, -6.28, -6.28])
        self.joint_maxs = np.array([6.28, 6.28, 3.14, 6.28, 6.28, 6.28])
        self.target_angles = np.zeros(6)

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

        self.step_count = 0

        self.data.qpos[:7] = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0])
        self.data.qvel[:7] = np.zeros(7)
        self.target_angles = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        
        distance = self.np_random.uniform(1.2, 1.8)
        angle = self.np_random.uniform(-np.pi/6, np.pi/6 )
        height_z = self.np_random.uniform(0.2, 0.8)
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

    def step(self, action):

        self.step_count += 1 

        scaled_action = np.zeros(7)
        self.target_angles += action[0:6] * 0.05
        self.target_angles = np.clip(self.target_angles, self.joint_mins, self.joint_maxs)

        scaled_action[:6] = self.target_angles
        scaled_action[6] = ((action[6] + 1.0) / 2.0) * 255.0
        
        self.data.ctrl[:7] = scaled_action

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Position ball-gripper
        ball_pos = self.data.body("target_object").xpos

        left_pad = self.data.geom("left_pad1").xpos
        right_pad = self.data.geom("right_pad1").xpos
        gripper_pos = (left_pad + right_pad) / 2.0

        #base to gripper distance
        base_distance = np.linalg.norm(gripper_pos - np.array([0.0, 0.0, 0.0]))
                
        #Velocity ball-gripper
        ball_vel = self.data.body("target_object").cvel[3:]
        gripper_vel = self.data.body("2f85_base").cvel[3:]
        ball_dir = ball_vel / (np.linalg.norm(ball_vel) + 1e-6)
        gripper_dir = gripper_vel / (np.linalg.norm(gripper_vel) + 1e-6)

        reward = 0.0
        terminated = False
        truncated = False 

        # Reward for getting close
        distance = np.linalg.norm(ball_pos - gripper_pos)
        reward += -distance *1.0

        # Penalty for moving too much
        reward -= np.sum(np.square(action[:6])) * 0.0001

        # Premature squeeze penalty
        if distance > 0.10 and action[6] > 0.0:
            reward -= 0.1

        # Penalty if the ball touches the ground
        if ball_pos[2] < 0.05:
            terminated = True
            reward -= 20.0 

        if base_distance < 0.30:
            reward -= 2.0

        if self.step_count > 1500:
            truncated = True

        # Reward for velocity match
        if distance < 0.15:
            velocity_match = np.dot(ball_dir, gripper_dir)
            reward += velocity_match * 0.6

        if distance < 0.06:
            if action[6] > 0.5:
                if ball_pos[2] > 0.2:
                    reward += 30.0
            else:
                reward -= 10.0

        info = self._get_info()

        return obs, reward, terminated, truncated, info
        
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

