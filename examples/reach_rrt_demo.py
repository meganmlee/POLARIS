"""
Visualize the RRT-Connect policy for Reach-WithObstacles-v1.

Runs a single episode using numerical IK and RRT-Connect planning
with live rendering.

Run:
    python examples/reach_rrt_demo.py
    python examples/reach_rrt_demo.py --seed 42
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
from reach_rrt import RRTConnect, solve_ik, smooth_path_spline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,   default=5)
    parser.add_argument("--render-fps", type=float, default=15.0,
                        help="Target rendering speed in frames per second")
    args = parser.parse_args()

    env = gym.make(
        "Reach-WithObstacles-v1",
        obs_mode="state_dict",        # required for IK state extraction
        control_mode="pd_joint_pos",  # required for joint-space execution
        render_mode="human",
        sim_backend="physx_cpu",      # CPU physx: set_qpos → FK updates immediately
    )
    root = env.unwrapped

    obs, _ = env.reset(seed=args.seed)
    env.render()

    q_start  = np.asarray(obs["agent"]["qpos"],     dtype=np.float32).reshape(-1)[:7]
    goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)

    print(f"Start configuration : {np.round(q_start, 3)}")
    print(f"Goal position (EE)  : {np.round(goal_pos, 3)}")
    print("Solving IK...")

    t0     = time.time()
    q_goal = solve_ik(root, goal_pos, q_start)

    if q_goal is None:
        print("IK failed to find a valid joint configuration. Exiting.")
        env.close()
        return

    print(f"IK solved in {time.time() - t0:.2f}s.  Goal config: {np.round(q_goal, 3)}")
    print("Planning trajectory with RRT-Connect...")

    t0      = time.time()
    planner = RRTConnect(q_start, q_goal)
    path    = planner.plan()
    print(f"RRT-Connect found path with {len(path)} waypoints in {time.time() - t0:.2f}s.")

    traj = smooth_path_spline(path, num_points=150)

    # pd_joint_pos action dim may include gripper joints beyond the 7 arm joints
    act_dim     = env.action_space.shape[0]
    gripper_pad = np.zeros(act_dim - 7, dtype=np.float32) + 0.04

    print("Executing trajectory...")

    for step, q in enumerate(traj):
        action = torch.tensor(np.concatenate([q, gripper_pad]), dtype=torch.float32).unsqueeze(0)
        obs, reward, terminated, truncated, info = env.step(action)
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
