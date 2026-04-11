"""
Closed-loop executor: generate subgoals → execute skill → re-plan → repeat.

Usage:
    python high_level_planner/executor.py --seed 42 --reach-checkpoint Reach-WithObstacles-v1__1__<timestamp>
    python high_level_planner/executor.py --seed 42 --skill mpc
    python high_level_planner/executor.py --seed 3 --max_replans 5 --offline --skill mpc
    python high_level_planner/executor.py --seed 0 \\
        --reach-checkpoint Reach-WithObstacles-v1__1__<ts> \\
        --push-cube-checkpoint PushCube-WithObstacles-v1__1__<ts>

Checkpoint args accept either a full path or just a run name from the checkpoints/ folder.
e.g. --reach-checkpoint Reach-WithObstacles-v1__1__1773025568
  expands to checkpoints/Reach-WithObstacles-v1__1__1773025568/final_ckpt.pt
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

_CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def _resolve_checkpoint(ckpt: str | None) -> str | None:
    """Accept a full path or just a run name — expand the latter to checkpoints/{name}/final_ckpt.pt."""
    if ckpt is None:
        return None
    p = Path(ckpt)
    if p.exists():
        return str(p)
    candidate = _CHECKPOINTS_DIR / ckpt / "final_ckpt.pt"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Checkpoint not found: {ckpt!r} (tried {candidate})")

import numpy as np

# Make project root and all skill dirs importable
_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "skills"))
sys.path.insert(0, os.path.join(_ROOT, "skills", "reach"))
sys.path.insert(0, os.path.join(_ROOT, "skills", "pick"))
sys.path.insert(0, os.path.join(_ROOT, "skills", "place"))
sys.path.insert(0, os.path.join(_ROOT, "skills", "push_cube"))

import envs  # noqa: F401 — registers ManiSkill envs
import gymnasium as gym
from planning_wrapper.adapters import PushTTaskAdapter
from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper

from llm_plan import region_to_xy
from env_subgoal_runner import subgoals_from_wrapper
from reach_mpc import execute as mpc_execute
from reach_ppo import execute as ppo_execute
from pick_cube_ppo import execute as pick_ppo_execute
from pick_cube_mpc import execute as pick_mpc_execute
from place_cube_ppo import execute as place_ppo_execute
from place_cube_mpc import execute as place_mpc_execute
from push_cube_ppo import execute as push_cube_ppo_execute
from push_cube_mpc import execute as push_cube_mpc_execute


def _parse_region(state_str: str) -> str | None:
    """Extract the region name from a PDDL atom, e.g. '(robot-at robot1 r_6_3)' → 'r_6_3'."""
    m = re.search(r"(r_\d+_\d+)", state_str)
    return m.group(1) if m else None


def _parse_block(state_str: str) -> int | None:
    """Extract obstacle index from a PDDL atom, e.g. '(holding robot1 obstacle3)' → 3."""
    m = re.search(r"obstacle(\d+)", state_str)
    return int(m.group(1)) if m else None


def run(
    seed: int = 0,
    max_replans: int = 10,
    offline: bool = False,
    model: str = "gemini-2.5-flash",
    render: bool = False,
    skill: str = "ppo",
    env_id: str = "PushT-WallObstacles-v1",
    checkpoint: str | None = None,
    pick_checkpoint: str | None = None,
    place_checkpoint: str | None = None,
    push_cube_checkpoint: str | None = None,
):
    control_mode = "pd_ee_delta_pose"
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode=control_mode,
        sim_backend="physx_cpu",
        render_mode="human" if render else None,
    )
    if render:
        _orig_render = env.render
        env.render = lambda *a, **kw: (_orig_render(*a, **kw), time.sleep(0.05))[0]

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
            block_idx = _parse_block(state)
            print(f"  {sg_skill}\t{state}")

            if sg_skill == "reach":
                if region is None:
                    print(f"    [WARN] could not parse region from: {state!r}")
                    skill_failed = True
                    break
                ee_z = float(
                    np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[2]
                )
                x, y = region_to_xy(region)
                goal_xyz = np.array([x, y, max(ee_z, 0.10)], dtype=np.float32)
                print(f"    → reach {np.round(goal_xyz, 3)}")

                if skill == "ppo":
                    success, obs = ppo_execute(env, obs, goal_xyz, checkpoint=checkpoint, render=render)
                else:
                    success, obs = mpc_execute(env, obs, goal_xyz, render=render)

                print(f"    → {'OK' if success else 'FAIL'}")
                if not success:
                    skill_failed = True
                    break

            elif sg_skill == "pick":
                if block_idx is None:
                    print(f"    [WARN] could not parse obstacle index from: {state!r}")
                    skill_failed = True
                    break
                print(f"    → pick cube{block_idx}")

                if skill == "ppo":
                    if pick_checkpoint is None:
                        print("    [SKIP] pick: no --pick-checkpoint provided — re-planning")
                        skill_failed = True
                        break
                    success, obs = pick_ppo_execute(
                        env, obs, block_idx, checkpoint=pick_checkpoint, render=render
                    )
                else:
                    success, obs = pick_mpc_execute(
                        env, obs, block_idx, render=render
                    )

                print(f"    → {'OK' if success else 'FAIL'}")
                if not success:
                    skill_failed = True
                    break

            elif sg_skill == "place":
                if block_idx is None or region is None:
                    print(f"    [WARN] could not parse block/region from: {state!r}")
                    skill_failed = True
                    break
                x, y = region_to_xy(region)
                goal_xyz = np.array([x, y, 0.02], dtype=np.float32)  # table-surface height
                print(f"    → place cube{block_idx} at {np.round(goal_xyz, 3)}")

                if skill == "ppo":
                    if place_checkpoint is None:
                        print("    [SKIP] place: no --place-checkpoint provided — re-planning")
                        skill_failed = True
                        break
                    success, obs = place_ppo_execute(
                        env, obs, block_idx, goal_xyz, checkpoint=place_checkpoint, render=render
                    )
                else:
                    success, obs = place_mpc_execute(
                        env, obs, block_idx, goal_xyz, render=render
                    )
                print(f"    → {'OK' if success else 'FAIL'}")
                if not success:
                    skill_failed = True
                    break

            elif sg_skill == "push_cube":
                if block_idx is None or region is None:
                    print(f"    [WARN] could not parse block/region from: {state!r}")
                    skill_failed = True
                    break
                x, y = region_to_xy(region)
                goal_xyz = np.array([x, y, 0.0], dtype=np.float32)
                print(f"    → push_cube obstacle{block_idx} to {np.round(goal_xyz, 3)}")

                if skill == "ppo":
                    if push_cube_checkpoint is None:
                        print("    [SKIP] push_cube: no --push-cube-checkpoint provided — re-planning")
                        skill_failed = True
                        break
                    success, obs = push_cube_ppo_execute(
                        env, obs, block_idx, goal_xyz, checkpoint=push_cube_checkpoint, render=render
                    )
                else:
                    success, obs = push_cube_mpc_execute(
                        env, obs, block_idx, goal_xyz, render=render
                    )
                print(f"    → {'OK' if success else 'FAIL'}")
                if not success:
                    skill_failed = True
                    break

            elif sg_skill == "push_tee":
                print(f"    [SKIP] 'push_tee' not yet implemented — re-planning")
                skill_failed = True
                break

            else:
                print(f"    [SKIP] '{sg_skill}' unknown — re-planning")
                skill_failed = True
                break

        if not skill_failed:
            print("All subgoals executed successfully.")
            break

    wrapper.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",                type=int,   default=0)
    ap.add_argument("--max_replans",         type=int,   default=10)
    ap.add_argument("--offline",             action="store_true")
    ap.add_argument("--render",              action="store_true", help="Open a viewer window")
    ap.add_argument("--model",               default="gemini-2.5-flash")
    ap.add_argument("--skill",               default="ppo", choices=["mpc", "ppo"])
    ap.add_argument("--reach-checkpoint",    default="Reach", dest="reach_checkpoint", help="Reach PPO checkpoint (required for --skill ppo)")
    ap.add_argument("--pick-checkpoint",     default="PickSkill", dest="pick_checkpoint",
                    help="Pick skill PPO checkpoint")
    ap.add_argument("--place-checkpoint",    default="PlaceSkill", dest="place_checkpoint",
                    help="Place skill PPO checkpoint")
    ap.add_argument("--push-cube-checkpoint", default="PushCube", dest="push_cube_checkpoint",
                    help="Push-cube skill PPO checkpoint")
    args = ap.parse_args()
    if args.skill == "ppo" and args.reach_checkpoint is None:
        ap.error("--reach-checkpoint is required when using --skill ppo")
    run(
        seed=args.seed,
        max_replans=args.max_replans,
        offline=args.offline,
        model=args.model,
        render=args.render,
        skill=args.skill,
        checkpoint=_resolve_checkpoint(args.reach_checkpoint),
        pick_checkpoint=_resolve_checkpoint(args.pick_checkpoint),
        place_checkpoint=_resolve_checkpoint(args.place_checkpoint),
        push_cube_checkpoint=_resolve_checkpoint(args.push_cube_checkpoint),
    )
