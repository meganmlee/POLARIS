"""
Visualize the RRT* policy for ReachGoal.

Runs a single episode using numerical IK and RRT* planning
with live rendering.

Run:
    python examples/reach_rrt_demo.py
"""

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import envs  # noqa: F401 — registers MoveGoal-WithObstacles-v1

# Import Planner and utils from reach_rrt
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "reach"))
from reach_rrt import RRTStar, solve_ik, smooth_path_spline # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--render-fps", type=float, default=15.0,
                        help="Target rendering speed in frames per second")
    args = parser.parse_args()

    # Note: RRT requires absolute joint positions, NOT ee_delta_pose
    env = gym.make(
        "ReachGoal",
        obs_mode="state_dict",         # Required for IK state extraction
        control_mode="pd_joint_pos",   # Required for joint-space execution
        render_mode="human",
    )
    root = env.unwrapped

    obs, _ = env.reset(seed=args.seed)
    env.render()

    q_start  = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)[:7]
    goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)
    
    print(f"Start configuration : {np.round(q_start, 3)}")
    print(f"Goal position (EE)  : {np.round(goal_pos, 3)}")
    print("Solving IK...")

    t0 = time.time()
    q_goal = solve_ik(root, goal_pos, q_start)
    
    if q_goal is None:
        print("IK failed to find a valid joint configuration. Exiting.")
        env.close()
        return
        
    print(f"IK Solved in {time.time() - t0:.2f}s. Goal config: {np.round(q_goal, 3)}")
    print("Planning trajectory with RRT*...")

    t0 = time.time()
    planner = RRTStar(q_start, q_goal)
    path = planner.plan()
    print(f"RRT* found path with {len(path)} waypoints in {time.time() - t0:.2f}s.")

    traj = smooth_path_spline(path, num_points=150)
    print("Executing trajectory...")

    for step, q in enumerate(traj):
        action = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        time.sleep(1.0 / args.render_fps)
        
        # Parse scalar or array success info
        success = info.get('success', False)
        if hasattr(success, 'item'):
            success = success.item()
        elif isinstance(success, (list, np.ndarray)):
            success = success[0]
            
        if success:
            print(f"Goal reached at step {step}!")
            break
            
        if terminated or truncated:
            print(f"Episode ended prematurely at step {step}.")
            break

    # Pause at the end before closing so you can see the final state
    time.sleep(2.0)
    env.close()

if __name__ == "__main__":
    main()