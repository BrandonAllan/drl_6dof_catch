# 🤖 Deep Reinforcement Learning for 6-DOF Robotic Catching

A research project implementing Deep Reinforcement Learning (DRL) to enable a 6-DOF robotic manipulator to autonomously catch falling objects in real-time. The system integrates trajectory prediction, dynamic motion planning, and learned manipulation policies to tackle the challenges of real-time grasping. The project combines physics simulation (PyBullet), advanced RL algorithms, and robotic control frameworks to develop a complete autonomous catching pipeline that bridges simulation and real-world deployment.

**Future Implementation**: Integration with ROS 2 Jazzy and MoveIt 2 for post-catch manipulation, enabling the robot to autonomously path-plan and place caught objects into designated containers.
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

## 🎓 Research Direction

This project addresses fundamental challenges in dynamic manipulation:
- **Real-time decision making** under uncertainty
- **Learning from trial and error** in physics-constrained environments
- **Generalization** to unseen object trajectories
- **Bridging simulation and reality** (sim-to-real transfer)

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