import argparse
import os
import re
import sys

import numpy as np

from llm_plan import GRID, TILE_SIZE_M, _xy_to_region, get_subgoals, state_to_problem


def subgoals_from_wrapper(wrapper, obs, model: str = "gemini-2.5-flash", temperature: float = 0.0, offline: bool = False, use_llm_first: bool = False):
    """
    Produce (PDDL_problem_str, subgoals) from an already-running wrapper+obs.
    Used by both run_from_env and the closed-loop executor.
    """
    planning_obs = wrapper.get_planning_obs(obs)

    tee_xy = np.asarray(planning_obs["obj_pose"], dtype=np.float64).reshape(-1, 7)[0, :2]
    goal_xy = np.asarray(planning_obs["goal_pos"], dtype=np.float64).reshape(-1, 3)[0, :2]
    ee_xy = np.asarray(planning_obs["tcp_pose"], dtype=np.float64).reshape(-1, 7)[0, :2]

    blocked = []
    root = wrapper.env.unwrapped
    if hasattr(root, "obstacles"):
        for obj in root.obstacles:
            pose = getattr(obj, "pose", None)
            if pose is None and callable(getattr(obj, "get_pose", None)):
                pose = obj.get_pose()
            if pose is not None:
                if hasattr(pose, "p"):
                    p = pose.p
                else:
                    try:
                        p = np.asarray(pose, dtype=np.float64).reshape(-1)[:3]
                    except (ValueError, TypeError):
                        continue
                pos = np.asarray(p, dtype=np.float64).reshape(-1)
                if len(pos) >= 2:
                    r = _xy_to_region(float(pos[0]), float(pos[1]))
                    if r not in blocked:
                        blocked.append(r)

    domain_path = os.path.join(os.path.dirname(__file__), "domain_pusht.pddl")
    problem_str = state_to_problem(tee_xy, goal_xy, ee_xy, blocked)
    subgoals = get_subgoals(
        domain_path,
        problem_str,
        model=model,
        temperature=temperature,
        offline=offline,
        use_llm_first=use_llm_first,
    )
    return problem_str, subgoals


def run_from_env(model: str = "gemini-2.5-flash", temperature: float = 0.0, offline: bool = False, use_llm_first: bool = False, seed: int = 0):
    """
    Produce (PDDL_problem_str, subgoals) using the live PushT-WithObstacles environment.

    Note: env imports stay inside this function so the planner module itself stays env-free.
    """
    import gymnasium as gym
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import envs  # noqa: F401
    from planning_wrapper.adapters import PushTTaskAdapter
    from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper

    env = gym.make(
        "PushT-WithObstacles-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
    )
    wrapper = ManiSkillPlanningWrapper(env, adapter=PushTTaskAdapter())
    obs, _ = wrapper.reset(seed=seed)
    problem_str, subgoals = subgoals_from_wrapper(wrapper, obs, model=model, temperature=temperature, offline=offline, use_llm_first=use_llm_first)
    wrapper.close()
    return problem_str, subgoals


def run_dummy(model: str = "gemini-2.5-flash", temperature: float = 0.0, offline: bool = False, use_llm_first: bool = False):
    """
    Produce (PDDL_problem_str, subgoals) using a dummy state (no env).
    Useful to sanity-check planner logic without ManiSkill running.
    """
    tee_xy = np.array([0.0, 0.0])       # → r_5_5
    goal_xy = np.array([0.09, -0.09])   # → r_3_6
    ee_xy = np.array([-0.02, 0.0])
    # Wall of 4 blocks across row 4 cols 4-7, directly between tee and goal.
    # Planner must pick one up rather than take a long detour.
    block_regions = [
        4 * GRID + 4,  # r_4_4
        4 * GRID + 5,  # r_4_5
        4 * GRID + 6,  # r_4_6
        4 * GRID + 7,  # r_4_7
    ]

    domain_path = os.path.join(os.path.dirname(__file__), "domain_pusht.pddl")
    problem_str = state_to_problem(tee_xy, goal_xy, ee_xy, block_regions)
    subgoals = get_subgoals(
        domain_path,
        problem_str,
        model=model,
        temperature=temperature,
        offline=offline,
        use_llm_first=use_llm_first,
    )
    return problem_str, subgoals


def _compact_state_summary(problem_str: str) -> str:
    robot = re.search(r"\(robot-at robot1 (r_\d+_\d+)\)", problem_str)
    tee = re.search(r"\(object-at tee (r_\d+_\d+)\)", problem_str)
    goal = re.search(r"\(goal-at (r_\d+_\d+)\)", problem_str)
    blocks = re.findall(r"\(block-at block\d+ (r_\d+_\d+)\)", problem_str)
    r = robot.group(1) if robot else "?"
    t = tee.group(1) if tee else "?"
    g = goal.group(1) if goal else "?"
    b = ", ".join(sorted(set(blocks))) if blocks else "none"
    return f"robot={r}  tee={t}  goal={g}  blocks=[{b}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Use live env state; else dummy state")
    ap.add_argument("--offline", action="store_true", help="No API: BFS subgoals only (no pick/place)")
    ap.add_argument("--llm", action="store_true", help="Skip PDDL planner; use LLM only (fallback path)")
    ap.add_argument("--verbose", action="store_true", help="Print state summary and full PDDL problem")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="Random seed for env reset (--live only)")
    args = ap.parse_args(argv)

    if args.live:
        problem_str, subgoals = run_from_env(
            model=args.model,
            temperature=args.temperature,
            offline=args.offline,
            use_llm_first=args.llm,
            seed=args.seed,
        )
    else:
        problem_str, subgoals = run_dummy(
            model=args.model,
            temperature=args.temperature,
            offline=args.offline,
            use_llm_first=args.llm,
        )

    if args.verbose:
        print("State:", _compact_state_summary(problem_str))
        if args.offline:
            print("Mode: offline (BFS; no pick/place)")
        elif args.llm:
            print("Mode: LLM only (planner skipped)")
        print(f"\nRegions: r_m_n, tile={TILE_SIZE_M:.4f}m\nProblem:\n{problem_str}\n")

    for sg in subgoals:
        print(f"{sg['skill']}\t{sg['state']}")


if __name__ == "__main__":
    main()

