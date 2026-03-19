"""
Skill: reach a goal EE position using RRT-Connect in 7-DOF joint space.

RRT-Connect grows two trees simultaneously (one from start, one from goal)
and is significantly faster and more reliable than RRT* for finding feasible
paths in joint space.

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

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import envs  # registers MoveGoal-WithObstacles-v1


JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


class RRTConnect:
    """
    Bidirectional RRT-Connect in 7-DOF joint space.

    Collision checking: joint limits only.
    """

    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        max_iter: int = 2000,
        step_size: float = 0.10,
        goal_threshold: float = 0.05,
    ):
        self.q_start = q_start.copy()
        self.q_goal  = q_goal.copy()
        self.max_iter       = max_iter
        self.step_size      = step_size
        self.goal_threshold = goal_threshold

        # Two trees: roots at q_start and q_goal respectively
        self.nodes_s   = [q_start.copy()]
        self.parents_s = [-1]
        self.nodes_g   = [q_goal.copy()]
        self.parents_g = [-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest(self, nodes: list, q: np.ndarray) -> int:
        return int(np.argmin([np.linalg.norm(n - q) for n in nodes]))

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(q_to - q_from)
        return q_to.copy() if d <= self.step_size else q_from + (q_to - q_from) / d * self.step_size

    def _valid(self, q: np.ndarray) -> bool:
        return bool(np.all((q >= JOINT_LOWER) & (q <= JOINT_UPPER)))

    def _extend(self, nodes: list, parents: list, q_target: np.ndarray):
        """Single-step extension toward q_target. Returns (q_new, new_idx) or (None, -1)."""
        idx   = self._nearest(nodes, q_target)
        q_new = self._steer(nodes[idx], q_target)
        if not self._valid(q_new):
            return None, -1
        nodes.append(q_new)
        parents.append(idx)
        return q_new, len(nodes) - 1

    def _connect(self, nodes: list, parents: list, q_target: np.ndarray) -> bool:
        """Greedily extend toward q_target until reached or blocked."""
        while True:
            q_new, _ = self._extend(nodes, parents, q_target)
            if q_new is None:
                return False
            if np.linalg.norm(q_new - q_target) <= self.step_size:
                return True

    def _tree_path(self, nodes: list, parents: list, leaf_idx: int) -> list[np.ndarray]:
        """Walk parent pointers from leaf to root, return root-to-leaf list."""
        path, idx = [], leaf_idx
        while idx != -1:
            path.append(nodes[idx].copy())
            idx = parents[idx]
        return path[::-1]

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def plan(self) -> list[np.ndarray]:
        """
        Returns a list of joint-space waypoints from q_start to q_goal.
        Falls back to the closest partial path if max_iter is exhausted.
        """
        for i in range(self.max_iter):
            q_rand = np.random.uniform(JOINT_LOWER, JOINT_UPPER)

            if i % 2 == 0:
                # Extend start-tree, then try to connect goal-tree
                q_new, idx_new = self._extend(self.nodes_s, self.parents_s, q_rand)
                if q_new is None:
                    continue
                if self._connect(self.nodes_g, self.parents_g, q_new):
                    path_s = self._tree_path(self.nodes_s, self.parents_s, idx_new)
                    path_g = self._tree_path(self.nodes_g, self.parents_g, len(self.nodes_g) - 1)
                    return path_s + path_g[::-1]
            else:
                # Extend goal-tree, then try to connect start-tree
                q_new, idx_new = self._extend(self.nodes_g, self.parents_g, q_rand)
                if q_new is None:
                    continue
                if self._connect(self.nodes_s, self.parents_s, q_new):
                    path_s = self._tree_path(self.nodes_s, self.parents_s, len(self.nodes_s) - 1)
                    path_g = self._tree_path(self.nodes_g, self.parents_g, idx_new)
                    return path_s + path_g[::-1]

        print("  [RRTConnect] max_iter reached — returning best partial path")
        idx_best = min(range(len(self.nodes_s)),
                       key=lambda i: np.linalg.norm(self.nodes_s[i] - self.q_goal))
        return self._tree_path(self.nodes_s, self.parents_s, idx_best)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

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
    Smooth an RRT path with a clamped cubic spline (zero velocity at endpoints).
    Returns a dense (num_points, 7) float32 trajectory clipped to joint limits.
    """
    if len(path) < 2:
        return np.array(path)

    path_arr = np.array(path)
    diffs = np.linalg.norm(np.diff(path_arr, axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(diffs)))
    if t[-1] > 0:
        t /= t[-1]

    cs = CubicSpline(t, path_arr, bc_type='clamped')
    smooth = cs(np.linspace(0, 1, num_points))
    return np.clip(smooth, JOINT_LOWER, JOINT_UPPER).astype(np.float32)


# ---------------------------------------------------------------------------
# Numerical IK
# ---------------------------------------------------------------------------

def solve_ik(
    root_env,
    goal_pos_world: np.ndarray,
    q0: np.ndarray,
    tol: float = 4e-4,
    max_restarts: int = 50,
) -> Optional[np.ndarray]:
    import sapien

    state = root_env.get_state_dict()

    full_q0     = root_env.agent.robot.qpos.cpu().numpy().reshape(-1)
    gripper_pad = full_q0[7:] if len(full_q0) > 7 else np.array([])

    pmodel = root_env.agent.robot.create_pinocchio_model()
    tcp_link_idx = 10  # panda_hand_tcp

    # Get current EE orientation to use as target orientation
    q_full_init = np.concatenate([q0, gripper_pad]).astype(np.float32)
    pmodel.compute_forward_kinematics(q_full_init)
    init_pose = pmodel.get_link_pose(tcp_link_idx)
    target_pose = sapien.Pose(p=goal_pos_world, q=init_pose.q)

    # Mask: only solve for the 7 arm joints, freeze gripper
    active_mask = np.zeros(len(full_q0), dtype=np.int32)
    active_mask[:7] = 1

    guesses = [q0] + [np.random.uniform(JOINT_LOWER, JOINT_UPPER).astype(np.float32) for _ in range(max_restarts)]
    best_q, best_residual = None, float("inf")

    for attempt, guess in enumerate(guesses):
        q_full = np.concatenate([guess, gripper_pad]).astype(np.float32)
        result_q, success, error = pmodel.compute_inverse_kinematics(
            tcp_link_idx,
            target_pose,
            initial_qpos=q_full,
            active_qmask=active_mask,
            max_iterations=1000,
        )

        if success:
            root_env.set_state_dict(state)
            if attempt > 0:
                print(f"  [IK] Solved on restart {attempt}")
            return np.clip(result_q[:7], JOINT_LOWER, JOINT_UPPER).astype(np.float32)

        if error < best_residual:
            best_residual = error
            best_q = result_q[:7]

    root_env.set_state_dict(state)
    print(f"  [IK] failed after {max_restarts + 1} attempts — best error {best_residual:.6f}")
    return None

# ---------------------------------------------------------------------------
# Callable skill API
# ---------------------------------------------------------------------------

def execute(
    env,
    obs: dict,
    goal_xyz: np.ndarray,
    max_iter: int = 2000,
    step_size: float = 0.10,
    goal_threshold: float = 0.05,
    num_traj_points: int = 150,
    render: bool = False,
) -> tuple[bool, dict]:
    """
    Move the EE to goal_xyz on an already-running env.

    Returns (success, latest_obs). Success is True if the EE lands within
    5 cm of the target after executing the trajectory.
    """
    root = env.unwrapped
    q_start = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)[:7]

    q_goal = solve_ik(root, goal_xyz, q_start)
    if q_goal is None:
        return False, obs

    planner = RRTConnect(q_start, q_goal, max_iter=max_iter, step_size=step_size, goal_threshold=goal_threshold)
    path = planner.plan()
    traj = smooth_path_spline(path, num_points=num_traj_points)

    act_dim = env.action_space.shape[0]
    gripper_pad = np.zeros(act_dim - 7, dtype=np.float32) + 0.04

    current_obs = obs
    for q in traj:
        action = torch.tensor(np.concatenate([q, gripper_pad]), dtype=torch.float32).unsqueeze(0)
        current_obs, _, term, trunc, _ = env.step(action)
        if render:
            env.render()
        if np.asarray(term).any() or np.asarray(trunc).any():
            break

    final_ee = np.asarray(current_obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    success = bool(np.linalg.norm(final_ee - goal_xyz) < 0.05)
    return success, current_obs


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    np.random.seed(args.seed)

    env = gym.make(
        "MoveGoal-WithObstacles-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reconfiguration_freq=1,
        sim_backend="physx_cpu",
    )
    root = env.unwrapped

    successes, final_dists, plan_times = [], [], []

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        obs, _ = env.reset()

        q_start  = np.asarray(obs["agent"]["qpos"],     dtype=np.float32).reshape(-1)[:7]
        goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)
        print(f"  q_start : {np.round(q_start, 3)}")
        print(f"  goal_pos: {np.round(goal_pos, 3)}")

        t0     = time.time()
        q_goal = solve_ik(root, goal_pos, q_start)

        if q_goal is None:
            successes.append(False); final_dists.append(float("inf")); plan_times.append(0.0)
            continue

        planner = RRTConnect(
            q_start, q_goal,
            max_iter=args.max_iter,
            step_size=args.step_size,
            goal_threshold=args.goal_threshold,
        )
        path = planner.plan()
        plan_times.append(time.time() - t0)
        print(f"  RRTConnect: {len(path)} waypoints  ({plan_times[-1]:.2f}s)")

        traj = smooth_path_spline(path, num_points=args.num_traj_points)

        # pd_joint_pos action dim may include gripper joints beyond the 7 arm joints
        act_dim     = env.action_space.shape[0]
        gripper_pad = np.zeros(act_dim - 7, dtype=np.float32) + 0.04

        success, final_dist = False, float("inf")
        for q in traj:
            action = torch.tensor(np.concatenate([q, gripper_pad]), dtype=torch.float32).unsqueeze(0)
            obs, _, term, trunc, info = env.step(action)

            dist = float(np.asarray(info.get("dist_to_goal", [float("inf")])).reshape(-1)[0])
            suc  = bool(np.asarray(info.get("success",      [False]       )).reshape(-1)[0])

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
    parser = argparse.ArgumentParser(description="RRT-Connect eval for MoveGoal-WithObstacles-v1")
    parser.add_argument("--num_episodes",    type=int,   default=10)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--max_iter",        type=int,   default=2000)
    parser.add_argument("--step_size",       type=float, default=0.10)
    parser.add_argument("--goal_threshold",  type=float, default=0.05)
    parser.add_argument("--num_traj_points", type=int,   default=150)
    args = parser.parse_args()
    run_eval(args)
