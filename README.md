# 🤖 Deep Reinforcement Learning for 6-DOF Robotic Catching

A research project implementing Deep Reinforcement Learning (DRL) to enable a 6-DOF robotic manipulator to autonomously catch falling objects in real-time. The system utilizes custom Delta Position Control to manage physical hardware limits. The project bridges advanced RL algorithms and physics simulations to tackle the challenges of high-speed dynamic grasping.

- Future Implementation: Integration with ROS 2 Jazzy and MoveIt 2 for post-catch manipulation, enabling the robot to autonomously path-plan and place caught objects into designated containers.
---

## 🎯 Project Scope

- 🧠 **DRL-based Control**: Training an agent using PPO via Stable Baselines3 for dynamic manipulation. 
- 🤖 **6-DOF Arm + Gripper**: Full kinematic chain control of a UR5e, including dynamic Tool Center Point (TCP) calculations for the 2F-85 gripper.
- 🎯 **Sim-to-Real Pipeline**: The neural network implicitly learns to track and intercept falling paths based on Cartesian coordinates and velocity vectors.
- 📦 **Multi-task Learning**: Catch → Transport → Place workflow (Phase 2).

---

## 🔧 Technologies

- **Python** – Core architecture and training scripts.
- **Mujoco** – Physics simulation and environment
- **Gymnasium** – Standardized RL environment API.
- **ROS 2 Jazzy & MoveIt 2** – Robotic middleware and motion planning (Planned Phase 2)

---

## 📋 Project Status

**Current Phase**: Phase 1 - DRL catching algorithm development
- ✅ Environment setup, CAD integration, and simulation.
- 🔄 Agent training and hyperparameter tuning (in progress).
- ⏳ ROS 2 + MoveIt 2 integration for post-catch manipulation (Phase 2).
- ⏳ Real robot hardware validation (Phase 3).

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install mujoco stable-baselines3 gymnasium numpy
```

### Running the Project
```bash
# To train a new model from scratch:
python3 src/train.py

# To watch the trained model perform in the 3D viewer:
python3 src/test.py
```