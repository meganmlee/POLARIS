"""
Closed-loop executor: generate subgoals → execute skill → re-plan → repeat.

Usage:
    python high_level_planner/executor.py --seed 42 --checkpoint runs/.../final_ckpt.pt
    python high_level_planner/executor.py --seed 42 --skill rrt
    python high_level_planner/executor.py --seed 3 --max_replans 5 --offline --skill rrt
"""
import argparse
import os
import re
import sys

import numpy as np

# Make project root and skills importable
_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "skills", "reach"))

import envs  # noqa: F401 — registers ManiSkill envs
import gymnasium as gym
from planning_wrapper.adapters import PushTTaskAdapter
from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper

from llm_plan import region_to_xy
from env_subgoal_runner import subgoals_from_wrapper
from reach_rrt import execute as rrt_execute
from reach_ppo import execute as ppo_execute


def _parse_region(state_str: str) -> str | None:
    """Extract the region name from a PDDL atom, e.g. '(robot-at robot1 r_6_3)' → 'r_6_3'."""
    m = re.search(r"(r_\d+_\d+)", state_str)
    return m.group(1) if m else None


def run(
    seed: int = 0,
    max_replans: int = 10,
    offline: bool = False,
    model: str = "gemini-2.5-flash",
    render: bool = False,
    skill: str = "ppo",
    checkpoint: str | None = None,
):
    control_mode = "pd_ee_delta_pose" if skill == "ppo" else "pd_joint_pos"
    env = gym.make(
        "PushT-WithObstacles-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode=control_mode,
        sim_backend="physx_cpu",
        render_mode="human" if render else None,
    )
    wrapper = ManiSkillPlanningWrapper(env, adapter=PushTTaskAdapter())
    obs, _ = wrapper.reset(seed=seed)

    for replan_i in range(max_replans):
        print(f"\n--- Plan {replan_i + 1}/{max_replans} ---")
        problem_str, subgoals = subgoals_from_wrapper(
            wrapper, obs, offline=offline, model=model
        )

        if not subgoals:
            print("No subgoals returned — task complete or unreachable.")
            break

        skill_failed = False
        for sg in subgoals:
            sg_skill = sg["skill"]
            state    = sg["state"]
            region   = _parse_region(state)
            print(f"  {sg_skill}\t{state}")

            if region is None:
                print(f"    [WARN] could not parse region from: {state!r}")
                continue

            if sg_skill == "reach":
                ee_z = float(
                    np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[2]
                )
                x, y = region_to_xy(region)
                goal_xyz = np.array([x, y, ee_z], dtype=np.float32)
                print(f"    → reach {np.round(goal_xyz, 3)}")

                if skill == "ppo":
                    success, obs = ppo_execute(env, obs, goal_xyz, checkpoint=checkpoint, render=render)
                else:
                    success, obs = rrt_execute(env, obs, goal_xyz, render=render)

                print(f"    → {'OK' if success else 'FAIL'}")
                if not success:
                    skill_failed = True
                    break

            else:
                print(f"    [SKIP] '{sg_skill}' not yet implemented — re-planning")
                skill_failed = True
                break

        if not skill_failed:
            print("All subgoals executed successfully.")
            break

    wrapper.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--max_replans", type=int,   default=10)
    ap.add_argument("--offline",     action="store_true")
    ap.add_argument("--render",      action="store_true", help="Open a viewer window")
    ap.add_argument("--model",       default="gemini-2.5-flash")
    ap.add_argument("--skill",       default="ppo", choices=["rrt", "ppo"])
    ap.add_argument("--checkpoint",  default=None, help="PPO checkpoint path (required for --skill ppo)")
    args = ap.parse_args()
    if args.skill == "ppo" and args.checkpoint is None:
        ap.error("--checkpoint is required when using --skill ppo")
    run(seed=args.seed, max_replans=args.max_replans, offline=args.offline,
        model=args.model, render=args.render, skill=args.skill, checkpoint=args.checkpoint)
