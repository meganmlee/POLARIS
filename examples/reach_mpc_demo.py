"""
Visualize the MPPI controller for Reach-WithObstacles-v1.

Runs a single episode using MPPI (Model Predictive Path Integral)
control with live rendering.

Run:
    python examples/reach_mpc_demo.py
    python examples/reach_mpc_demo.py --seed 42
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import numpy as np
import torch

import envs  # noqa: F401 — registers Reach-WithObstacles-v1

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "reach"))
from reach_mpc import MPPI  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,   default=5)
    parser.add_argument("--render-fps", type=float, default=15.0,
                        help="Target rendering speed in frames per second")
    parser.add_argument("--max-steps",  type=int,   default=200)
    args = parser.parse_args()

    env = gym.make(
        "Reach-WithObstacles-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        sim_backend="physx_cpu",
    )

    obs, _ = env.reset(seed=args.seed)
    env.render()

    goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)
    ee_pos   = np.asarray(obs["extra"]["ee_pos"],  dtype=np.float32).reshape(-1)[:3]

    print(f"Start EE position  : {np.round(ee_pos, 3)}")
    print(f"Goal position (EE) : {np.round(goal_pos, 3)}")
    print("Running MPPI controller...")

    controller = MPPI(goal_xyz=goal_pos)
    act_dim = env.action_space.shape[0]

    for step in range(args.max_steps):
        ee_pos = np.asarray(obs["extra"]["ee_pos"], dtype=np.float32).reshape(-1)[:3]
        delta_pos = controller.get_action(ee_pos)

        action = np.zeros(act_dim, dtype=np.float32)
        action[:3] = delta_pos
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        obs, reward, terminated, truncated, info = env.step(action_t)
        env.render()
        time.sleep(1.0 / args.render_fps)

        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()
        elif isinstance(success, (list, np.ndarray)):
            success = bool(success[0])

        if success:
            print(f"Goal reached at step {step}!")
            break
        if terminated or truncated:
            print(f"Episode ended at step {step}.")
            break

    time.sleep(2.0)
    env.close()


if __name__ == "__main__":
    main()
