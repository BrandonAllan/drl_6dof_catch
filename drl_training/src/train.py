import mujoco
from stable_baselines3 import PPO
from enviroment import CathRobotEnv

env = CathRobotEnv()
print("Initializing AI Brain...")
model = PPO("MlpPolicy", env, verbose=1)

print("Starting Training! (Press Ctrl+C to stop at any time)")

model.learn(total_timesteps=4000000)

model.save("drl_training/results/robot_weights.zip")

print("Training complete and model saved!")
