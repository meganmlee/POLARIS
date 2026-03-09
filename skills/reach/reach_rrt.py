"""
Skill: reach a goal EE position using RRT* in 7-DOF joint space.

IK is solved numerically with the env's SAPIEN FK as an oracle + scipy
L-BFGS-B — no external kinematics library required.

Control: pd_joint_pos (absolute joint position setpoints).

Usage:
    python skills/reach/reach_rrt.py
    python skills/reach/reach_rrt.py --num_episodes 20 --seed 42
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

import envs  # registers ReachGoal


JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


class RRTStar:
    """RRT* in 7-DOF joint space. Collision checking = joint limits only."""

    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        max_iter: int = 1500,
        step_size: float = 0.15,
        goal_sample_rate: float = 0.10,
        goal_threshold: float = 0.05,
        search_radius: float = 0.3,
    ):
        self.q_goal           = q_goal.copy()
        self.max_iter         = max_iter
        self.step_size        = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_threshold   = goal_threshold
        self.search_radius    = search_radius
        
        self.nodes            = [q_start.copy()]
        self.parents          = [-1]
        self.costs            = [0.0]

    def _sample(self) -> np.ndarray:
        if np.random.rand() < self.goal_sample_rate:
            return self.q_goal.copy()
        return np.random.uniform(JOINT_LOWER, JOINT_UPPER)

    def _nearest(self, q: np.ndarray) -> int:
        return int(np.argmin([np.linalg.norm(n - q) for n in self.nodes]))

    def _near_indices(self, q: np.ndarray) -> list[int]:
        return [i for i, n in enumerate(self.nodes) if np.linalg.norm(n - q) <= self.search_radius]

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(q_to - q_from)
        return q_to.copy() if d <= self.step_size else q_from + (q_to - q_from) / d * self.step_size

    def plan(self) -> list[np.ndarray]:
        for _ in range(self.max_iter):
            q_rand   = self._sample()
            idx_near = self._nearest(q_rand)
            q_new    = self._steer(self.nodes[idx_near], q_rand)

            if not np.all((q_new >= JOINT_LOWER) & (q_new <= JOINT_UPPER)):
                continue

            near_indices = self._near_indices(q_new)
            
            # Connect along a minimum-cost path
            min_cost = self.costs[idx_near] + np.linalg.norm(q_new - self.nodes[idx_near])
            min_cost_idx = idx_near

            for idx in near_indices:
                cost = self.costs[idx] + np.linalg.norm(q_new - self.nodes[idx])
                if cost < min_cost:
                    min_cost = cost
                    min_cost_idx = idx

            self.nodes.append(q_new)
            self.parents.append(min_cost_idx)
            self.costs.append(min_cost)
            new_node_idx = len(self.nodes) - 1

            # Rewire the tree
            for idx in near_indices:
                if idx == min_cost_idx:
                    continue
                rewired_cost = self.costs[new_node_idx] + np.linalg.norm(self.nodes[idx] - q_new)
                if rewired_cost < self.costs[idx]:
                    self.parents[idx] = new_node_idx
                    self.costs[idx] = rewired_cost

            if np.linalg.norm(q_new - self.q_goal) < self.goal_threshold:
                return self._extract_path(len(self.nodes) - 1)

        print("  [RRT*] max_iter reached — returning best partial path")
        idx_best = min(range(len(self.nodes)), key=lambda i: np.linalg.norm(self.nodes[i] - self.q_goal))
        return self._extract_path(idx_best)

    def _extract_path(self, leaf: int) -> list[np.ndarray]:
        path, idx = [], leaf
        while idx != -1:
            path.append(self.nodes[idx].copy())
            idx = self.parents[idx]
        return path[::-1]


def interpolate_path(path: list[np.ndarray], steps_per_segment: int = 15) -> np.ndarray:
    """Linear interpolation between RRT waypoints → dense (T, 7) trajectory."""
    traj = []
    for i in range(len(path) - 1):
        for a in np.linspace(0, 1, steps_per_segment, endpoint=False):
            traj.append(path[i] + a * (path[i + 1] - path[i]))
    traj.append(path[-1])
    return np.array(traj)

def smooth_path_spline(path: list[np.ndarray], num_points: int = 150) -> np.ndarray:
    """
    Smooths an RRT* path using a Cubic Spline.
    Ensures zero velocity at the start and end of the trajectory.
    """
    if len(path) < 2:
        return np.array(path)

    path_arr = np.array(path)  # Shape: (N, 7)
    
    # Calculate cumulative distance along the path for time-parameterization
    diffs = np.linalg.norm(np.diff(path_arr, axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(diffs)))
    
    # Normalize t to [0, 1] to represent overall progress
    if t[-1] > 0:
        t /= t[-1]
        
    # Fit a cubic spline. 
    # bc_type='clamped' forces the first derivative (velocity) to 0 at the start and end.
    cs = CubicSpline(t, path_arr, bc_type='clamped')
    
    # Generate the dense, smooth trajectory
    t_smooth = np.linspace(0, 1, num_points)
    smooth_traj = cs(t_smooth)
    
    # Clip to ensure no minor overshoot violates joint limits
    smooth_traj = np.clip(smooth_traj, JOINT_LOWER, JOINT_UPPER)
    
    return smooth_traj.astype(np.float32)


def solve_ik(
    root_env,
    goal_pos_world: np.ndarray,
    q0: np.ndarray,
    tol: float = 4e-4,
    max_restarts: int = 10,  # Number of random restarts to try
) -> Optional[np.ndarray]:
    """
    Numerical IK using the env's SAPIEN FK as an oracle.
    Uses random restarts to avoid local minima.
    """
    device = root_env.device
    state  = root_env.get_state_dict()

    def cost(q: np.ndarray) -> float:
        q_t = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
        root_env.agent.robot.set_qpos(q_t)
        ee = root_env.agent.tcp.pose.p.cpu().numpy().reshape(-1)
        return float(np.sum((ee - goal_pos_world) ** 2))

    # Queue of initial guesses: try the robot's current pose first
    guesses = [q0]
    for _ in range(max_restarts):
        guesses.append(np.random.uniform(JOINT_LOWER, JOINT_UPPER))

    best_residual = float("inf")

    for attempt, guess in enumerate(guesses):
        result = minimize(
            cost, guess,
            method="L-BFGS-B",
            bounds=list(zip(JOINT_LOWER, JOINT_UPPER)),
            options={"maxiter": 300, "ftol": 1e-10},
        )
        
        if result.fun < best_residual:
            best_residual = result.fun
            
        if result.fun < tol:
            root_env.set_state_dict(state)  # always restore the sim state
            if attempt > 0:
                print(f"  [IK] Solved on random restart {attempt}")
            return np.clip(result.x, JOINT_LOWER, JOINT_UPPER).astype(np.float32)

    root_env.set_state_dict(state)
    print(f"  [IK] failed after {max_restarts + 1} attempts — best residual {best_residual:.6f} > tol {tol}")
    return None

def run_eval(args):
    np.random.seed(args.seed)

    env = gym.make(
        "ReachGoal",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reconfiguration_freq=1,
    )
    root = env.unwrapped

    successes, final_dists, plan_times = [], [], []

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        obs, _ = env.reset()

        q_start  = np.asarray(obs["agent"]["qpos"],  dtype=np.float32).reshape(-1)[:7]
        goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)
        print(f"  q_start : {np.round(q_start, 3)}")
        print(f"  goal_pos: {np.round(goal_pos, 3)}")

        t0     = time.time()
        q_goal = solve_ik(root, goal_pos, q_start)

        if q_goal is None:
            successes.append(False); final_dists.append(float("inf")); plan_times.append(0.0)
            continue

        planner = RRTStar(q_start, q_goal,
                      max_iter=args.max_iter,
                      step_size=args.step_size,
                      goal_sample_rate=args.goal_sample_rate,
                      goal_threshold=args.goal_threshold)
        path = planner.plan()
        plan_times.append(time.time() - t0)
        print(f"  RRT: {len(path)} waypoints  ({plan_times[-1]:.2f}s)")

        traj = interpolate_path(path, steps_per_segment=args.steps_per_segment)

        success, final_dist = False, float("inf")
        for q in traj:
            action = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
            obs, _, term, trunc, info = env.step(action)

            dist = float(np.asarray(info.get("dist_to_goal", [float("inf")])).reshape(-1)[0])
            suc  = bool(np.asarray(info.get("success", [False])).reshape(-1)[0])

            final_dist = dist
            if suc:
                success = True
                break
            if np.asarray(term).any() or np.asarray(trunc).any():
                break

        successes.append(success)
        final_dists.append(final_dist)
        print(f"  {'SUCCESS' if success else 'FAIL'}  dist={final_dist * 100:.1f} cm")

    env.close()

    valid = [d for d in final_dists if d < float("inf")]
    print("\n" + "=" * 50)
    print(f"Success rate   : {np.mean(successes) * 100:.1f}%")
    print(f"Mean final dist: {np.mean(valid) * 100:.1f} cm" if valid else "Mean final dist: N/A")
    print(f"Mean plan time : {np.mean(plan_times):.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RRT eval for ReachGoal")
    parser.add_argument("--num_episodes",      type=int,   default=10)
    parser.add_argument("--seed",              type=int,   default=0)
    parser.add_argument("--max_iter",          type=int,   default=1000)
    parser.add_argument("--step_size",         type=float, default=0.15)
    parser.add_argument("--goal_sample_rate",  type=float, default=0.10)
    parser.add_argument("--goal_threshold",    type=float, default=0.05)
    parser.add_argument("--steps_per_segment", type=int,   default=15)
    args = parser.parse_args()
    run_eval(args)
