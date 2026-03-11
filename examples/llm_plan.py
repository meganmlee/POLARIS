import json
import os
import re
import subprocess
import tempfile
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", message=".*deprecated.*")

# Prefer PDDL planner for correct, generalizable subgoals; LLM only as fallback.
PLANNER_PRIORITY = True

GRID = 10
TABLE_BOUND = 0.12
TILE_SIZE_M = (2.0 * TABLE_BOUND) / GRID  # each tile side length in m; change GRID or TABLE_BOUND to change it

def _region_to_name(idx: int) -> str:
    row, col = idx // GRID, idx % GRID
    return f"r_{row}_{col}"

def _name_to_idx(name: str) -> int:
    m = re.match(r"r_(\d+)_(\d+)", name.strip())
    if not m:
        return 0
    row, col = int(m.group(1)), int(m.group(2))
    return row * GRID + col

def _xy_to_region(x: float, y: float) -> int:
    s = TABLE_BOUND
    cx = int((float(x) + s) / (2 * s) * GRID)
    cy = int((float(y) + s) / (2 * s) * GRID)
    cx = max(0, min(GRID - 1, cx))
    cy = max(0, min(GRID - 1, cy))
    return cy * GRID + cx

def _adjacent_pairs():
    out = []
    for i in range(GRID):
        for j in range(GRID):
            r = i * GRID + j
            if j + 1 < GRID:
                out.append((r, r + 1))
            if j - 1 >= 0:
                out.append((r, r - 1))
            if i + 1 < GRID:
                out.append((r, r + GRID))
            if i - 1 >= 0:
                out.append((r, r - GRID))
    return out

NUM_BLOCKS = 10
NUM_PICKABLE = 5

def _region_layout_comment() -> str:
    return (
        f"; REGIONS: r_m_n with m=row (0..{GRID-1}), n=column (0..{GRID-1}). "
        f"Each tile physical size: {TILE_SIZE_M:.4f} m x {TILE_SIZE_M:.4f} m (set GRID and TABLE_BOUND in code to change). "
        "Adjacent = share an edge (e.g. r_2_2 adjacent to r_1_2, r_2_1, r_2_3, r_3_2)."
    )

def state_to_problem(tee_xy: np.ndarray, goal_xy: np.ndarray, ee_xy: np.ndarray, block_regions: list) -> str:
    rt = _xy_to_region(tee_xy[0], tee_xy[1])
    rg = _xy_to_region(goal_xy[0], goal_xy[1])
    re = _xy_to_region(ee_xy[0], ee_xy[1])
    blocks = list(block_regions) if len(block_regions) >= NUM_BLOCKS else list(block_regions) + [0] * (NUM_BLOCKS - len(block_regions))
    blocks = blocks[:NUM_BLOCKS]
    region_set = set(blocks)
    region_names = [_region_to_name(i) for i in range(GRID * GRID)]
    objects = "robot1 - robot tee - tee " + " ".join(f"block{i}" for i in range(NUM_BLOCKS)) + " - block " + " ".join(region_names) + " - region"
    init = [
        f"(robot-at robot1 {_region_to_name(re)})",
        f"(object-at tee {_region_to_name(rt)})",
        f"(goal-at {_region_to_name(rg)})",
    ]
    for i in range(NUM_PICKABLE):
        init.append(f"(pickable block{i})")
    for i in range(NUM_PICKABLE, NUM_BLOCKS):
        init.append(f"(push-only block{i})")
    for i, r in enumerate(blocks):
        init.append(f"(block-at block{i} {_region_to_name(r)})")
    for i in range(GRID * GRID):
        if i not in region_set:
            init.append(f"(clear {_region_to_name(i)})")
    for a, b in _adjacent_pairs():
        init.append(f"(adjacent {_region_to_name(a)} {_region_to_name(b)})")
    goal = f"(object-at tee {_region_to_name(rg)})"
    path_tee_to_goal_cells = set(_bfs_path(rt, rg, set()))
    blocks_on_path = [i for i in range(NUM_BLOCKS) if blocks[i] in path_tee_to_goal_cells]
    block_list = ", ".join(f"block{i}@{_region_to_name(blocks[i])}" for i in range(NUM_BLOCKS))
    on_path_str = (
        f" Blocks ON THE PATH from tee to goal (MUST clear first): " + ", ".join(f"block{i} at {_region_to_name(blocks[i])}" for i in blocks_on_path) + "."
    ) if blocks_on_path else " No blocks on the direct path; you may still need to clear some if the only route goes past them."
    scenario = (
        f"; State: robot1={_region_to_name(re)}, tee={_region_to_name(rt)}, goal={_region_to_name(rg)}. Blocks: {block_list}. "
        "block0..block4 are pickable (use pick then place to move off path). block5..block9 are push-only (use push_block)."
        f"{on_path_str} "
        "Your output must be ONLY a sequence of subgoals: each line SKILL then TAB then STATE (one PDDL atom in parentheses)."
    )
    return f"""; PushT with obstacles: get the T-shaped object (tee) to the goal region.
{_region_layout_comment()}
{scenario}
(define (problem pusht-1)
  (:domain pusht)
  (:objects {objects})
  (:init
    {" ".join(init)}
  )
  (:goal {goal})
)"""

def _load_config():
    for path in [
        os.path.join(os.path.dirname(__file__), "config.json"),
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
    ]:
        if os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
    return {}

def _parse_problem_regions(problem_str: str):
    robot = re.search(r"\(robot-at robot1 (r_\d+_\d+)\)", problem_str)
    tee = re.search(r"\(object-at tee (r_\d+_\d+)\)", problem_str)
    goal = re.search(r"\(goal-at (r_\d+_\d+)\)", problem_str)
    blocked = set(_name_to_idx(m.group(1)) for m in re.finditer(r"\(block-at block\d+ (r_\d+_\d+)\)", problem_str))
    return (
        _name_to_idx(tee.group(1)) if tee else 0,
        _name_to_idx(goal.group(1)) if goal else 0,
        _name_to_idx(robot.group(1)) if robot else 0,
        blocked,
    )

def _adjacency():
    adj = [[] for _ in range(GRID * GRID)]
    for a, b in _adjacent_pairs():
        adj[a].append(b)
    return adj

def _bfs_path(start: int, goal: int, blocked: set) -> list:
    from collections import deque
    adj = _adjacency()
    seen = {start}
    q = deque([(start, [])])
    while q:
        r, path = q.popleft()
        if r == goal:
            return path + [r]
        for n in adj[r]:
            if n not in seen and n not in blocked:
                seen.add(n)
                q.append((n, path + [r]))
    return []

def compute_subgoals(problem_str: str) -> list[dict]:
    """BFS-based subgoals (move_ee to tee, then push_tee to goal). Returns list of {skill, state}."""
    tee_r, goal_r, robot_r, blocked = _parse_problem_regions(problem_str)
    path_robot_to_tee = _bfs_path(robot_r, tee_r, blocked)
    path_tee_to_goal = _bfs_path(tee_r, goal_r, blocked)
    subgoals = []
    if path_robot_to_tee and path_robot_to_tee[-1] == tee_r:
        subgoals.append({"skill": "move_ee", "state": f"(robot-at robot1 {_region_to_name(tee_r)})"})
    for i in range(1, len(path_tee_to_goal)):
        subgoals.append({"skill": "push_tee", "state": f"(object-at tee {_region_to_name(path_tee_to_goal[i])})"})
    if not subgoals and path_tee_to_goal and path_tee_to_goal[0] != goal_r:
        subgoals.append({"skill": "push_tee", "state": f"(object-at tee {_region_to_name(goal_r)})"})
    return subgoals


def _push_tee_subgoals_only(problem_str: str) -> list[dict]:
    """BFS path from tee to goal as push_tee subgoals only."""
    tee_r, goal_r, _r, blocked = _parse_problem_regions(problem_str)
    path = _bfs_path(tee_r, goal_r, blocked)
    if len(path) < 2:
        return []
    return [{"skill": "push_tee", "state": f"(object-at tee {_region_to_name(path[i])})"} for i in range(1, len(path))]


def run_pddl_planner(domain_path: str, problem_str: str, timeout: int = 60) -> str | None:
    """Run a PDDL planner (pyperplan if available). Returns plan string or None if no solution/failure."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as f:
        f.write(problem_str)
        problem_path = f.name
    try:
        subprocess.run(
            ["pyperplan", domain_path, problem_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        soln_path = problem_path + ".soln"
        if os.path.isfile(soln_path):
            with open(soln_path, "r") as f:
                plan = f.read().strip()
            try:
                os.remove(soln_path)
            except OSError:
                pass
            return plan if plan else None
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            os.remove(problem_path)
        except OSError:
            pass


def plan_to_subgoals(plan_str: str, _problem_str: str) -> list[dict]:
    """Convert a PDDL plan (one action per line) to subgoals [{skill, state}]. Deterministic, no LLM."""
    subgoals = []
    for line in plan_str.splitlines():
        line = line.strip()
        if not line or not line.startswith("("):
            continue
        m = re.match(r"\(move_ee robot1 r_\d+_\d+ (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "move_ee", "state": f"(robot-at robot1 {m.group(1)})"})
            continue
        m = re.match(r"\(push_tee robot1 r_\d+_\d+ (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "push_tee", "state": f"(object-at tee {m.group(1)})"})
            continue
        m = re.match(r"\(pick robot1 (block\d+) r_\d+_\d+\)", line)
        if m:
            subgoals.append({"skill": "pick", "state": f"(holding robot1 {m.group(1)})"})
            continue
        m = re.match(r"\(place robot1 (block\d+) (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "place", "state": f"(block-at {m.group(1)} {m.group(2)})"})
            continue
        m = re.match(r"\(push_block robot1 (block\d+) r_\d+_\d+ (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "push_block", "state": f"(block-at {m.group(1)} {m.group(2)})"})
    return subgoals


VALID_SKILLS = frozenset({"move_ee", "push_tee", "pick", "place", "push_block"})


def _parse_subgoals_response(text: str) -> list[dict]:
    """Parse LLM output into list of {skill, state}. Expects lines: SKILL\tSTATE or SKILL STATE."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if not line or line.startswith(";"):
            continue
        if "\t" in line:
            skill, _, state = line.partition("\t")
        else:
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            skill, state = parts[0].strip(), parts[1].strip()
        skill = skill.lower().strip()
        state = state.strip()
        if state.startswith("(") and ")" in state and skill in VALID_SKILLS:
            out.append({"skill": skill, "state": state})
    return out


def _goal_region_from_problem(problem_str: str) -> str | None:
    m = re.search(r"\(goal-at (r_\d+_\d+)\)", problem_str)
    return m.group(1) if m else None


def _call_gemini_subgoals(domain: str, problem_str: str, model: str, temperature: float, config: dict) -> list[dict]:
    project = os.environ.get("VERTEX_PROJECT") or config.get("VERTEX_PROJECT")
    location = os.environ.get("VERTEX_LOCATION") or config.get("VERTEX_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("Set VERTEX_PROJECT in env or config.json for Gemini")
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except ImportError:
        raise RuntimeError("Install google-cloud-aiplatform: pip install google-cloud-aiplatform")
    vertexai.init(project=project, location=location)
    _default_gemini = "gemini-2.5-flash"
    gemini_model = model if (model and not model.startswith("gpt-")) else _default_gemini
    if gemini_model.startswith("gpt-"):
        gemini_model = _default_gemini
    gen_model = GenerativeModel(gemini_model)
    goal_region = _goal_region_from_problem(problem_str) or "r_0_0"
    prompt = f"""You are a task planner. Output a COMPLETE sequence of subgoals to get the TEE to the goal. Output ONLY subgoals, one per line: SKILL<TAB>STATE.

CRITICAL: The goal region in this problem is {goal_region}. Your sequence MUST end with:
push_tee	(object-at tee {goal_region})
So you MUST include a full chain of push_tee subgoals (one per cell) from the tee's current region to {goal_region}. Do not stop at move_ee or after one push_tee.

SKILLS (exactly one per line):
- move_ee → state (robot-at robot1 r_X_Y)
- push_tee → state (object-at tee r_X_Y)   [use many of these: one per step toward goal]
- pick → state (holding robot1 blockN)   [only for block0..block4]
- place → state (block-at blockN r_X_Y)
- push_block → state (block-at blockN r_X_Y)   [only for block5..block9]

REQUIRED ORDER:
1. If the problem comment says "Blocks ON THE PATH (MUST clear first): blockN at r_...", then FIRST output subgoals to clear each: move_ee to that block's region, then for block0..block4: pick then place (block at a clear region off path); for block5..block9: push_block (one cell off path).
2. Then move_ee so the robot is at the TEE's region (state: (robot-at robot1 r_TEE)).
3. Then a SEQUENCE of push_tee subgoals: each line push_tee	(object-at tee r_X_Y) for the next region along the path from tee to goal. The LAST line must be push_tee	(object-at tee {goal_region}).

Format: each line is SKILL then TAB then STATE. State is one PDDL atom in parentheses. No other text.

Example (goal r_1_8, tee at r_5_5):
move_ee	(robot-at robot1 r_2_5)
pick	(holding robot1 block0)
place	(block-at block0 r_2_6)
move_ee	(robot-at robot1 r_5_5)
push_tee	(object-at tee r_4_5)
push_tee	(object-at tee r_3_5)
push_tee	(object-at tee r_2_5)
push_tee	(object-at tee r_1_5)
push_tee	(object-at tee r_1_6)
push_tee	(object-at tee r_1_7)
push_tee	(object-at tee r_1_8)

Domain:
{domain}

Problem:
{problem_str}

Subgoals (one per line, SKILL<TAB>STATE; must end with push_tee	(object-at tee {goal_region})):
"""
    response = gen_model.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": 2048},
    )
    if not response.candidates or not response.candidates[0].content.parts:
        raise RuntimeError("Gemini returned empty response")
    text = response.candidates[0].content.parts[0].text.strip()
    return _parse_subgoals_response(text)

def get_subgoals(domain_path: str, problem_str: str, model: str = "gemini-2.5-flash", temperature: float = 0.0, offline: bool = False, use_llm_first: bool = False) -> list[dict]:
    """Return sequence of subgoals [{skill, state}, ...]. By default: PDDL planner first (robust); LLM only as fallback."""
    if offline or os.environ.get("LLM_PLAN_OFFLINE"):
        return compute_subgoals(problem_str)
    if PLANNER_PRIORITY and not use_llm_first:
        plan = run_pddl_planner(domain_path, problem_str)
        if plan:
            return plan_to_subgoals(plan, problem_str)
    with open(domain_path, "r") as f:
        domain = f.read()
    config = _load_config()
    subgoals = _call_gemini_subgoals(domain, problem_str, model, temperature, config)
    if not subgoals:
        return compute_subgoals(problem_str)
    goal_region = _goal_region_from_problem(problem_str)
    goal_state = f"(object-at tee {goal_region})" if goal_region else None
    has_push_tee = any(s.get("skill") == "push_tee" for s in subgoals)
    ends_at_goal = subgoals and subgoals[-1].get("state") == goal_state
    if goal_region and (not has_push_tee or not ends_at_goal):
        push_only = _push_tee_subgoals_only(problem_str)
        if push_only:
            subgoals = subgoals + push_only
    return subgoals

def run_from_env(model: str = "gemini-2.5-flash", temperature: float = 0.0, offline: bool = False, use_llm_first: bool = False) -> tuple[str, list[dict]]:
    import gymnasium as gym
    import pusht_w_obstacles
    from planning_wrapper.adapters import PushTTaskAdapter
    from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper

    env = gym.make(
        "PushT-WithObstacles-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
    )
    adapter = PushTTaskAdapter()
    wrapper = ManiSkillPlanningWrapper(env, adapter=adapter)
    obs, _ = wrapper.reset(seed=0)
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
                p = getattr(pose, "p", np.asarray(pose)[:3])
                pos = np.asarray(p, dtype=np.float64).reshape(-1)
                if len(pos) >= 2:
                    r = _xy_to_region(float(pos[0]), float(pos[1]))
                    if r not in blocked:
                        blocked.append(r)
    domain_path = os.path.join(os.path.dirname(__file__), "domain_pusht.pddl")
    problem_str = state_to_problem(tee_xy, goal_xy, ee_xy, blocked)
    subgoals = get_subgoals(domain_path, problem_str, model=model, temperature=temperature, offline=offline, use_llm_first=use_llm_first)
    wrapper.close()
    return problem_str, subgoals

def run_dummy(model: str = "gemini-2.5-flash", temperature: float = 0.0, offline: bool = False, use_llm_first: bool = False) -> tuple[str, list[dict]]:
    tee_xy = np.array([0.0, 0.0])
    goal_xy = np.array([0.08, -0.08])
    ee_xy = np.array([-0.02, 0.0])
    n = GRID * GRID
    block_regions = [0, 1, 2, 3, 4, n // 4, n // 2, 3 * n // 4, n - 8, n - 1]
    domain_path = os.path.join(os.path.dirname(__file__), "domain_pusht.pddl")
    problem_str = state_to_problem(tee_xy, goal_xy, ee_xy, block_regions)
    subgoals = get_subgoals(domain_path, problem_str, model=model, temperature=temperature, offline=offline, use_llm_first=use_llm_first)
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Use live env state; else dummy state")
    ap.add_argument("--offline", action="store_true", help="No API: BFS subgoals only (no pick/place)")
    ap.add_argument("--llm", action="store_true", help="Skip PDDL planner; use LLM only (fallback path)")
    ap.add_argument("--verbose", action="store_true", help="Print state summary and full PDDL problem")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    if args.live:
        problem_str, subgoals = run_from_env(model=args.model, temperature=args.temperature, offline=args.offline, use_llm_first=args.llm)
    else:
        problem_str, subgoals = run_dummy(model=args.model, temperature=args.temperature, offline=args.offline, use_llm_first=args.llm)
    if args.verbose:
        print("State:", _compact_state_summary(problem_str))
        if args.offline:
            print("Mode: offline (BFS; no pick/place)")
        elif args.llm:
            print("Mode: LLM only (planner skipped)")
        print(f"\nRegions: r_m_n, tile={TILE_SIZE_M:.4f}m\nProblem:\n{problem_str}\n")
    for sg in subgoals:
        print(f"{sg['skill']}\t{sg['state']}")
