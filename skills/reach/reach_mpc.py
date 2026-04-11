"""
Skill: reach a goal EE position using MPPI (Model Predictive Path Integral)
control in task space.

MPPI (Williams et al., 2016-2017) is a sampling-based MPC algorithm:
  1. Sample K random action sequences over horizon H (Gaussian perturbations).
  2. Roll each out through dynamics to get K trajectories.
  3. Compute a scalar cost per trajectory.
  4. Weight samples by exp(-cost / lambda) -- low-cost trajectories dominate.
  5. Weighted average of action sequences -> new nominal sequence.
  6. Execute first action. Shift nominal forward. Repeat.

Control: pd_ee_delta_pose (Cartesian delta position + orientation + gripper).

Usage:
    python skills/reach/reach_mpc.py
    python skills/reach/reach_mpc.py --num_episodes 20 --seed 42
"""
from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import envs  # registers Reach-WithObstacles-v1


class MPPI:
    """
    Model Predictive Path Integral controller for Cartesian EE reaching.

    Actions are 3-D EE delta positions. The dynamics model is a simple
    integrator: next_pos = current_pos + action (the PD controller tracks).
    """

    def __init__(
        self,
        goal_xyz: np.ndarray,
        horizon: int = 8,
        num_samples: int = 256,
        noise_std: float = 0.02,
        lam: float = 0.05,
        cost_goal_weight: float = 1.0,
        cost_control_weight: float = 0.01,
    ):
        self.goal_xyz = goal_xyz.copy()
        self.horizon = horizon
        self.K = num_samples
        self.noise_std = noise_std
        self.lam = lam
        self.w_goal = cost_goal_weight
        self.w_ctrl = cost_control_weight

        # Nominal action sequence: (H, 3) -- initialised to zeros
        self.nominal = np.zeros((horizon, 3), dtype=np.float32)

    def _rollout_costs(self, ee_pos: np.ndarray, action_seqs: np.ndarray) -> np.ndarray:
        """
        Simple forward model: pos_{t+1} = pos_t + action_t.
        Returns cost array of shape (K,).
        """
        K, H, _ = action_seqs.shape
        pos = np.broadcast_to(ee_pos, (K, 3)).copy()  # (K, 3)
        costs = np.zeros(K, dtype=np.float32)

        for t in range(H):
            pos = pos + action_seqs[:, t, :]
            dist = np.linalg.norm(pos - self.goal_xyz, axis=1)
            ctrl = np.sum(action_seqs[:, t, :] ** 2, axis=1)
            costs += self.w_goal * dist + self.w_ctrl * ctrl

        # Terminal cost: heavier penalty on final distance
        final_dist = np.linalg.norm(pos - self.goal_xyz, axis=1)
        costs += self.w_goal * 5.0 * final_dist

        return costs

    def get_action(self, ee_pos: np.ndarray) -> np.ndarray:
        """
        Run one MPPI step: sample, evaluate, reweight, return first action.
        """
        noise = np.random.randn(self.K, self.horizon, 3).astype(np.float32) * self.noise_std
        action_seqs = self.nominal[None, :, :] + noise  # (K, H, 3)

        costs = self._rollout_costs(ee_pos, action_seqs)

        # MPPI weighting: exp(-cost / lambda)
        costs_shifted = costs - np.min(costs)
        weights = np.exp(-costs_shifted / self.lam)
        weights /= np.sum(weights) + 1e-10

        # Weighted average -> updated nominal
        self.nominal = np.einsum("k,kha->ha", weights, action_seqs)

        action = self.nominal[0].copy()

        # Shift nominal forward (warm-start next step), pad last with zeros
        self.nominal = np.roll(self.nominal, -1, axis=0)
        self.nominal[-1] = 0.0

        return action


def _get_ee_pos(obs: dict) -> np.ndarray:
    """Extract EE position from either raw env obs or planning wrapper obs."""
    if "extra" in obs:
        # Raw env obs: obs["extra"]["ee_pos"] or obs["extra"]["tcp_pose"]
        extra = obs["extra"]
        if "ee_pos" in extra:
            return np.asarray(extra["ee_pos"], dtype=np.float32).reshape(-1)[:3]
        return np.asarray(extra["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    # Planning wrapper obs: top-level "tcp_pose"
    return np.asarray(obs["tcp_pose"], dtype=np.float32).reshape(-1)[:3]


# ---------------------------------------------------------------------------
# Callable skill API
# ---------------------------------------------------------------------------

def execute(
    env,
    obs: dict,
    goal_xyz: np.ndarray,
    max_steps: int = 200,
    horizon: int = 8,
    num_samples: int = 256,
    noise_std: float = 0.02,
    lam: float = 0.05,
    success_threshold: float = 0.05,
    render: bool = False,
    **kwargs,
) -> tuple[bool, dict]:
    """
    Move the EE to goal_xyz on an already-running env using MPPI.

    Returns (success, latest_obs). Success is True if the EE lands within
    5 cm of the target.
    """
    controller = MPPI(
        goal_xyz=goal_xyz,
        horizon=horizon,
        num_samples=num_samples,
        noise_std=noise_std,
        lam=lam,
    )

    act_dim = env.action_space.shape[0]
    current_obs = obs

    for step in range(max_steps):
        ee_pos = _get_ee_pos(current_obs)

        dist = np.linalg.norm(ee_pos - goal_xyz)
        if dist < success_threshold:
            return True, current_obs

        delta_pos = controller.get_action(ee_pos)

        # pd_ee_delta_pose: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = np.zeros(act_dim, dtype=np.float32)
        action[:3] = delta_pos

        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        current_obs, _, term, trunc, _ = env.step(action_t)
        if render:
            env.render()
        if np.asarray(term).any() or np.asarray(trunc).any():
            break

    final_ee = _get_ee_pos(current_obs)
    success = bool(np.linalg.norm(final_ee - goal_xyz) < success_threshold)
    return success, current_obs


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    np.random.seed(args.seed)

    env = gym.make(
        "Reach-WithObstacles-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        reconfiguration_freq=1,
        sim_backend="physx_cpu",
    )

    successes, final_dists = [], []

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        obs, _ = env.reset()

        ee_pos   = np.asarray(obs["extra"]["ee_pos"], dtype=np.float32).reshape(-1)[:3]
        goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)
        print(f"  ee_start: {np.round(ee_pos, 3)}")
        print(f"  goal_pos: {np.round(goal_pos, 3)}")

        t0 = time.time()
        success, final_obs = execute(
            env, obs, goal_pos,
            max_steps=args.max_steps,
            horizon=args.horizon,
            num_samples=args.num_samples,
            noise_std=args.noise_std,
            lam=args.lam,
        )
        elapsed = time.time() - t0

        final_ee = np.asarray(final_obs["extra"]["ee_pos"], dtype=np.float32).reshape(-1)[:3]
        dist = float(np.linalg.norm(final_ee - goal_pos))

        successes.append(success)
        final_dists.append(dist)
        print(f"  {'SUCCESS' if success else 'FAIL'}  dist={dist * 100:.1f} cm  ({elapsed:.2f}s)")

    env.close()

    valid = [d for d in final_dists if d < float("inf")]
    print("\n" + "=" * 50)
    print(f"Success rate   : {np.mean(successes) * 100:.1f}%")
    print(f"Mean final dist: {np.mean(valid) * 100:.1f} cm" if valid else "Mean final dist: N/A")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPPI eval for Reach-WithObstacles-v1")
    parser.add_argument("--num_episodes", type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--max_steps",    type=int,   default=200)
    parser.add_argument("--horizon",      type=int,   default=8)
    parser.add_argument("--num_samples",  type=int,   default=256)
    parser.add_argument("--noise_std",    type=float, default=0.02)
    parser.add_argument("--lam",          type=float, default=0.05)
    args = parser.parse_args()
    run_eval(args)
