"""
Weight scheme comparison test for skills/metrics.py

For each seed, at a reach decision point:
  1. Run the MPC and PPO lookahead previews once — capture (m_n, p_n, c_n) for each
  2. Execute both backends to completion to get ground truth (which one actually succeeds)
  3. Re-score each backend under every weight scheme using the same captured triplets
  4. Record whether each scheme picked the actual winner

Accuracy = fraction of decisions where the scheme chose the winning backend.

Usage:
    python tests/test_weight_schemes.py --checkpoint <run_name> --seeds 10

The test skips seeds where both backends succeed or both fail (no signal).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "skills"))
sys.path.insert(0, str(_ROOT / "skills" / "reach"))

import envs  # noqa: F401
import gymnasium as gym
from planning_wrapper.adapters import PushOTaskAdapter
from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from ppo_base import load_agent
from reach_mpc import execute as mpc_execute, ReachMPPI
from reach_ppo import _build_flat_obs, execute as ppo_execute
from mpc_base import get_ee_pos
from skills.metrics import (
    lookahead_rollout_score,
    lookahead_reach_mppi_score,
    lookahead_rl_score,
    weighted_reach_score,
    _normalize_triplet,
)

# ---------------------------------------------------------------------------
# Weight schemes to compare
# ---------------------------------------------------------------------------

def _equal(m_n, p_n, c_n):
    return 1/3 * m_n + 1/3 * p_n + 1/3 * c_n

def _progress_heavy(m_n, p_n, c_n):
    return 0.15 * m_n + 0.50 * p_n + 0.35 * c_n

def _clearance_heavy(m_n, p_n, c_n):
    return 0.15 * m_n + 0.35 * p_n + 0.50 * c_n

def _dynamic(m_n, p_n, c_n):
    """
    Weights derived from the normalized values themselves:
      w_prog  ∝ (1 - p_n)       — high when little progress was made (far from goal)
      w_clear ∝ 1/(c_n * 0.25 + 0.05) — high when near obstacle
      w_manip ∝ 1/(m_n * 0.25 + 0.05) — high when near singularity
    """
    est_manip = m_n * 0.25
    est_clear = c_n * 0.25
    w_p = max(1 - p_n, 1e-3)
    w_c = 1.0 / (est_clear + 0.05)
    w_m = 1.0 / (est_manip + 0.05)
    total = w_p + w_c + w_m
    return (w_m / total) * m_n + (w_p / total) * p_n + (w_c / total) * c_n

SCHEMES: dict[str, Any] = {
    "equal (1/3 each)":       _equal,
    "progress-heavy (0.5/0.35/0.15)": _progress_heavy,
    "clearance-heavy (0.5/0.35/0.15)": _clearance_heavy,
    "dynamic":                _dynamic,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reach_policy_act(goal_xyz, agent, env, device):
    action_low  = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)
    g = goal_xyz.astype(np.float32)
    def policy_act(obs):
        flat  = _build_flat_obs(obs, g)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)
    return policy_act


def _score_triplet(fn, info: dict) -> float:
    return fn(info["manip_norm"], info["progress_norm"], info["clearance_norm"])


def _pick_backend(fn, mpc_info: dict, ppo_info: dict) -> str:
    return "mpc" if _score_triplet(fn, mpc_info) >= _score_triplet(fn, ppo_info) else "ppo"


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------

def _final_dist(obs: dict, goal_xyz: np.ndarray) -> float:
    """EE distance to goal from a state_dict obs."""
    ee = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    return float(np.linalg.norm(ee - goal_xyz))


def evaluate_seed(
    seed: int,
    goal_xyz: np.ndarray,
    checkpoint: str,
    env_id: str,
    device: str,
    preview_steps: int,
) -> dict | None:
    """
    Returns a result dict, or None on error.

    Ground truth = which backend reached a lower final EE distance to goal.
    Using continuous distance instead of binary success means we always get signal,
    even when both backends succeed (one may do it faster / more cleanly).
    Seeds where both backends end up within 1mm of each other are skipped as ties.
    """
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu",
        render_mode=None,
    )
    wrapper = ManiSkillPlanningWrapper(env, adapter=PushOTaskAdapter())
    obs, _ = wrapper.reset(seed=seed)
    agent   = load_agent(checkpoint, device)

    # ---- 1. Capture preview triplets for both backends ----
    policy_act = _reach_policy_act(goal_xyz, agent, env, device)

    mpc_score, mpc_info = lookahead_reach_mppi_score(
        wrapper, obs, goal_xyz, preview_steps=preview_steps
    )
    ppo_score, ppo_info = lookahead_rl_score(
        wrapper, goal_xyz, policy_act, obs, preview_steps=preview_steps
    )

    # ---- 2. Ground truth: run each backend from the SAME starting state ----
    snap = wrapper.clone_state()

    _, obs_after_mpc = mpc_execute(env, obs, goal_xyz, render=False)
    mpc_dist = _final_dist(obs_after_mpc, goal_xyz)
    wrapper.restore_state(snap)

    _, obs_after_ppo = ppo_execute(
        env, obs, goal_xyz, checkpoint=checkpoint, render=False, device=device, agent=agent
    )
    ppo_dist = _final_dist(obs_after_ppo, goal_xyz)
    wrapper.restore_state(snap)

    wrapper.close()

    # Skip near-ties — difference < 1 mm is noise
    if abs(mpc_dist - ppo_dist) < 1e-3:
        return None

    ground_truth = "mpc" if mpc_dist < ppo_dist else "ppo"

    # ---- 3. Score each scheme ----
    picks = {name: _pick_backend(fn, mpc_info, ppo_info) for name, fn in SCHEMES.items()}
    correct = {name: (pick == ground_truth) for name, pick in picks.items()}

    return {
        "seed": seed,
        "ground_truth": ground_truth,
        "mpc_dist": mpc_dist,
        "ppo_dist": ppo_dist,
        "mpc_score_equal": mpc_score,
        "ppo_score_equal": ppo_score,
        "mpc_info": mpc_info,
        "ppo_info": ppo_info,
        "picks": picks,
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Reach",
                    help="Reach PPO checkpoint name or path")
    ap.add_argument("--seeds", type=int, default=20,
                    help="Number of seeds to evaluate")
    ap.add_argument("--preview-steps", type=int, default=24)
    ap.add_argument("--env-id", default="PushO-WallObstacles-v1")
    ap.add_argument("--device", default=None)
    # Fixed reach goals that exercise different parts of the workspace
    ap.add_argument("--goal", nargs=3, type=float, default=None,
                    help="Override goal XYZ (otherwise cycles through presets)")
    args = ap.parse_args()

    _CHECKPOINTS = _ROOT / "checkpoints"
    ckpt = args.checkpoint
    if not Path(ckpt).exists():
        candidate = _CHECKPOINTS / ckpt / "final_ckpt.pt"
        if candidate.exists():
            ckpt = str(candidate)
        else:
            ap.error(f"Checkpoint not found: {ckpt!r}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Preset goals that put the robot in varied situations:
    # near obstacles, clear space, long reach, short reach
    PRESET_GOALS = [
        np.array([0.10,  0.20, 0.15], dtype=np.float32),  # moderate, clear
        np.array([-0.10, 0.30, 0.15], dtype=np.float32),  # left side
        np.array([0.25,  0.05, 0.20], dtype=np.float32),  # far right
        np.array([0.00,  0.35, 0.10], dtype=np.float32),  # far forward, low
    ]

    results = []
    skipped = 0

    for seed in range(args.seeds):
        goal_xyz = (
            np.array(args.goal, dtype=np.float32)
            if args.goal
            else PRESET_GOALS[seed % len(PRESET_GOALS)]
        )
        print(f"seed={seed:3d}  goal={np.round(goal_xyz, 3)} ...", end=" ", flush=True)
        try:
            r = evaluate_seed(seed, goal_xyz, ckpt, args.env_id, device, args.preview_steps)
        except Exception as e:
            print(f"ERROR: {e}")
            skipped += 1
            continue

        if r is None:
            print("skip (both same outcome)")
            skipped += 1
            continue

        gt    = r["ground_truth"]
        picks = r["picks"]
        tag   = "  ".join(f"{n[:8]}={'OK' if c else '--'}" for n, c in r["correct"].items())
        mpc_i, ppo_i = r["mpc_info"], r["ppo_info"]
        triplet = (f"mpc(M:{mpc_i['manip_norm']:.2f} P:{mpc_i['progress_norm']:.2f} C:{mpc_i['clearance_norm']:.2f} d={r['mpc_dist']:.3f}) "
                   f"ppo(M:{ppo_i['manip_norm']:.2f} P:{ppo_i['progress_norm']:.2f} C:{ppo_i['clearance_norm']:.2f} d={r['ppo_dist']:.3f})")
        print(f"truth={gt}  {tag}  |  {triplet}")
        results.append(r)

    if not results:
        print("\nNo informative seeds — adjust goals or increase --seeds.")
        return

    # ---- Summary table ----
    n = len(results)
    print(f"\n{'='*60}")
    print(f"Results over {n} informative seeds  ({skipped} skipped)\n")
    print(f"{'Scheme':<40} {'Accuracy':>8}  {'Correct/Total':>14}")
    print(f"{'-'*40} {'-'*8}  {'-'*14}")

    scheme_scores: dict[str, list[bool]] = {name: [] for name in SCHEMES}
    for r in results:
        for name in SCHEMES:
            scheme_scores[name].append(r["correct"][name])

    for name in SCHEMES:
        correct_list = scheme_scores[name]
        acc = sum(correct_list) / len(correct_list)
        print(f"{name:<40} {acc:>7.1%}  {sum(correct_list):>6}/{len(correct_list):<6}")

    print(f"\nGround truth breakdown: "
          f"mpc wins {sum(r['ground_truth']=='mpc' for r in results)}, "
          f"ppo wins {sum(r['ground_truth']=='ppo' for r in results)}")

    # Per-scheme breakdown: when did they disagree from equal-weights, and were they right?
    print(f"\nDisagreements vs equal-weights baseline:")
    equal_name = "equal (1/3 each)"
    for name in SCHEMES:
        if name == equal_name:
            continue
        disagree = [r for r in results if r["picks"][name] != r["picks"][equal_name]]
        if not disagree:
            print(f"  {name}: never disagreed")
            continue
        right_on_disagree = sum(r["correct"][name] for r in disagree)
        print(f"  {name}: disagreed {len(disagree)}x, was right {right_on_disagree}/{len(disagree)}")


if __name__ == "__main__":
    main()
