"""
PushT-with-obstacles subgoal planning.

Order: Gemini proposes logical steps → optional repair against a full symbolic plan
(pyperplan) so the sequence is sound → else full PDDL plan → else BFS.
Output: list of {"skill", "state"}.
"""
import json
import os
import re
import subprocess
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*deprecated.*")

GRID = 10
TABLE_BOUND = 0.30
TILE_SIZE_M = (2.0 * TABLE_BOUND) / GRID

NUM_OBSTACLES = 10
NUM_PICKABLE = 5

VALID_SKILLS = frozenset({"reach", "push_tee", "pick", "place", "push_cube"})


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


def region_to_xy(name: str) -> tuple[float, float]:
    """Center of a grid tile in world XY (metres). Inverse of _xy_to_region."""
    m = re.match(r"r_(\d+)_(\d+)", name.strip())
    if not m:
        return 0.0, 0.0
    row, col = int(m.group(1)), int(m.group(2))
    x = (col + 0.5) / GRID * 2 * TABLE_BOUND - TABLE_BOUND
    y = (row + 0.5) / GRID * 2 * TABLE_BOUND - TABLE_BOUND
    return float(x), float(y)


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


def _region_layout_comment() -> str:
    return (
        f"; REGIONS: r_m_n, m row 0..{GRID - 1}, n col 0..{GRID - 1}. "
        f"Tile size ~{TILE_SIZE_M:.4f} m. Adjacent = edge neighbors."
    )


def state_to_problem(tee_xy: np.ndarray, goal_xy: np.ndarray, ee_xy: np.ndarray, obstacle_regions: list) -> str:
    rt = _xy_to_region(tee_xy[0], tee_xy[1])
    rg = _xy_to_region(goal_xy[0], goal_xy[1])
    re = _xy_to_region(ee_xy[0], ee_xy[1])
    obstacles = list(obstacle_regions)[:NUM_OBSTACLES]
    n_obstacles = len(obstacles)
    n_pickable = min(NUM_PICKABLE, n_obstacles)
    region_set = set(obstacles)
    region_names = [_region_to_name(i) for i in range(GRID * GRID)]
    objects = "robot1 - robot tee - tee " + " ".join(f"obstacle{i}" for i in range(n_obstacles)) + " - obstacle " + " ".join(region_names) + " - region"
    init = [
        f"(robot-at robot1 {_region_to_name(re)})",
        f"(object-at tee {_region_to_name(rt)})",
        f"(goal-at {_region_to_name(rg)})",
    ]
    for i in range(n_pickable):
        init.append(f"(pickable obstacle{i})")
    for i in range(n_pickable, n_obstacles):
        init.append(f"(push-only obstacle{i})")
    for i, r in enumerate(obstacles):
        init.append(f"(obstacle-at obstacle{i} {_region_to_name(r)})")
    for i in range(GRID * GRID):
        if i not in region_set:
            init.append(f"(clear {_region_to_name(i)})")
    for a, b in _adjacent_pairs():
        init.append(f"(adjacent {_region_to_name(a)} {_region_to_name(b)})")
    goal = f"(object-at tee {_region_to_name(rg)})"
    path_tee_to_goal_cells = set(_bfs_path(rt, rg, set()))
    obstacles_on_path = [i for i in range(n_obstacles) if obstacles[i] in path_tee_to_goal_cells]
    obstacle_list = ", ".join(f"obstacle{i}@{_region_to_name(obstacles[i])}" for i in range(n_obstacles))
    on_path_str = (
        f" Obstacles ON THE PATH (clear first): " + ", ".join(f"obstacle{i} at {_region_to_name(obstacles[i])}" for i in obstacles_on_path) + "."
    ) if obstacles_on_path else " No obstacles on direct tee→goal path; may still need clears."
    pickable_range = f"obstacle0..{n_pickable - 1}" if n_pickable > 0 else "none"
    pushonly_range = f"obstacle{n_pickable}..{n_obstacles - 1}" if n_pickable < n_obstacles else "none"
    scenario = (
        f"; robot={_region_to_name(re)}, tee={_region_to_name(rt)}, goal={_region_to_name(rg)}. Obstacles: {obstacle_list}. "
        f"{pickable_range} pickable (pick+place); {pushonly_range} push-only (push_cube). "
        f"{on_path_str} "
        "Output: one line per subgoal, SKILL<TAB>STATE (one PDDL atom in parens)."
    )
    return f"""; PushT with obstacles — tee to goal.
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
    blocked = set(_name_to_idx(m.group(1)) for m in re.finditer(r"\(obstacle-at obstacle\d+ (r_\d+_\d+)\)", problem_str))
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


def _parse_obstacle_info(problem_str: str) -> tuple[dict, set, set]:
    """Parse obstacle positions, pickable set, and push-only set from a problem string."""
    positions = {
        m.group(1): _name_to_idx(m.group(2))
        for m in re.finditer(r"\(obstacle-at (obstacle\d+) (r_\d+_\d+)\)", problem_str)
    }
    pickable  = {m.group(1) for m in re.finditer(r"\(pickable (obstacle\d+)\)", problem_str)}
    push_only = {m.group(1) for m in re.finditer(r"\(push-only (obstacle\d+)\)", problem_str)}
    return positions, pickable, push_only


def _clear_path_subgoals(tee_r: int, goal_r: int, blocked: set, problem_str: str) -> list[dict]:
    """
    When no tee→goal path exists around blocks, generate pick/place or push_cube
    subgoals to clear the blocks sitting on the direct (unblocked) tee→goal route.
    After these subgoals the executor re-plans, at which point a free path should exist.
    """
    direct_path = _bfs_path(tee_r, goal_r, set())
    if not direct_path:
        return []

    path_cells = set(direct_path)
    positions, pickable, push_only = _parse_obstacle_info(problem_str)
    adj = _adjacency()

    subgoals = []
    for obstacle_name, obstacle_region in positions.items():
        if obstacle_region not in path_cells:
            continue

        # Preferred drop zone: adjacent to obstacle, not on path, not already blocked
        drop = next(
            (n for n in adj[obstacle_region] if n not in path_cells and n not in blocked),
            next((n for n in adj[obstacle_region] if n not in blocked), None),
        )

        subgoals.append({
            "skill": "reach",
            "state": f"(robot-at robot1 {_region_to_name(obstacle_region)})",
        })

        if obstacle_name in pickable:
            subgoals.append({"skill": "pick",  "state": f"(holding robot1 {obstacle_name})"})
            if drop is not None:
                subgoals.append({"skill": "place", "state": f"(obstacle-at {obstacle_name} {_region_to_name(drop)})"})
        elif obstacle_name in push_only and drop is not None:
            subgoals.append({"skill": "push_cube", "state": f"(obstacle-at {obstacle_name} {_region_to_name(drop)})"})

    return subgoals


def compute_subgoals(problem_str: str) -> list[dict]:
    tee_r, goal_r, robot_r, blocked = _parse_problem_regions(problem_str)
    path_robot_to_tee = _bfs_path(robot_r, tee_r, blocked)
    path_tee_to_goal  = _bfs_path(tee_r, goal_r, blocked)
    subgoals = []
    if path_robot_to_tee and path_robot_to_tee[-1] == tee_r:
        subgoals.append({"skill": "reach", "state": f"(robot-at robot1 {_region_to_name(tee_r)})"})
    if path_tee_to_goal:
        for i in range(1, len(path_tee_to_goal)):
            subgoals.append({"skill": "push_tee", "state": f"(object-at tee {_region_to_name(path_tee_to_goal[i])})"})
    else:
        # All routes blocked — emit clearing subgoals for blocks on the direct path,
        # then append push_tee steps using a BFS path with those blocks removed.
        subgoals.extend(_clear_path_subgoals(tee_r, goal_r, blocked, problem_str))
        direct_path = _bfs_path(tee_r, goal_r, set())
        cleared_cells = set(direct_path) & blocked
        new_path = _bfs_path(tee_r, goal_r, blocked - cleared_cells)
        if new_path and len(new_path) >= 2:
            subgoals.append({"skill": "reach", "state": f"(robot-at robot1 {_region_to_name(tee_r)})"})
            for i in range(1, len(new_path)):
                subgoals.append({"skill": "push_tee", "state": f"(object-at tee {_region_to_name(new_path[i])})"})
    if not subgoals and path_tee_to_goal and path_tee_to_goal[0] != goal_r:
        subgoals.append({"skill": "push_tee", "state": f"(object-at tee {_region_to_name(goal_r)})"})
    return subgoals


def _push_tee_subgoals_only(problem_str: str) -> list[dict]:
    tee_r, goal_r, _, blocked = _parse_problem_regions(problem_str)
    path = _bfs_path(tee_r, goal_r, blocked)
    if len(path) < 2:
        return []
    return [{"skill": "push_tee", "state": f"(object-at tee {_region_to_name(path[i])})"} for i in range(1, len(path))]


def run_pddl_planner(domain_path: str, problem_str: str, timeout: int = 60) -> str | None:
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
    subgoals = []
    for line in plan_str.splitlines():
        line = line.strip()
        if not line or not line.startswith("("):
            continue
        m = re.match(r"\(reach robot1 r_\d+_\d+ (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "reach", "state": f"(robot-at robot1 {m.group(1)})"})
            continue
        m = re.match(r"\(push_tee robot1 r_\d+_\d+ (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "push_tee", "state": f"(object-at tee {m.group(1)})"})
            continue
        m = re.match(r"\(pick robot1 (obstacle\d+) r_\d+_\d+\)", line)
        if m:
            subgoals.append({"skill": "pick", "state": f"(holding robot1 {m.group(1)})"})
            continue
        m = re.match(r"\(place robot1 (obstacle\d+) (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "place", "state": f"(obstacle-at {m.group(1)} {m.group(2)})"})
            continue
        m = re.match(r"\(push_cube robot1 (obstacle\d+) r_\d+_\d+ (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "push_cube", "state": f"(obstacle-at {m.group(1)} {m.group(2)})"})
    return subgoals


def _parse_subgoals_response(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if not line or line.startswith(";"):
            continue
        m = re.match(r"Subgoal\s*\d+\s*:\s*(.+)", line, re.I)
        if m:
            line = m.group(1).strip()
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
    default_model = "gemini-2.5-flash"
    gemini_model = model if model and not model.startswith("gpt-") else default_model
    if gemini_model.startswith("gpt-"):
        gemini_model = default_model
    gen_model = GenerativeModel(gemini_model)
    goal_region = _goal_region_from_problem(problem_str) or "r_0_0"
    prompt = f"""You decompose this manipulation task into ordered logical subgoals (milestones), like splitting a big goal into steps.
Each line is one milestone: the SKILL that achieves it, then TAB, then the target state as ONE PDDL atom in parentheses.

Rules:
- Order matters: earlier lines must be achievable before later ones.
- End with: push_tee	(object-at tee {goal_region})
- Before that push chain: if obstacles block the path, clear them (pick/place or push_cube), then reach to the tee's cell, then many push_tee steps (one grid cell per push_tee toward {goal_region}).
- Skills: reach, push_tee, pick, place, push_cube. States: (robot-at robot1 r_i_j), (object-at tee r_i_j), (holding robot1 obstacleN), (obstacle-at obstacleN r_i_j).

Domain:
{domain}

Problem:
{problem_str}

Output ONLY lines: SKILL<TAB>STATE. No other text."""
    response = gen_model.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": 2048},
    )
    if not response.candidates or not response.candidates[0].content.parts:
        raise RuntimeError("Gemini returned empty response")
    text = response.candidates[0].content.parts[0].text.strip()
    return _parse_subgoals_response(text)


def _align_gemini_to_symbolic(gemini_steps: list[dict], symbolic_steps: list[dict]) -> list[dict]:
    """Sound order from symbolic plan; keep Gemini's skill when a step's state matches."""
    if not symbolic_steps:
        return gemini_steps
    out = []
    j = 0
    for g in gemini_steps:
        while j < len(symbolic_steps) and symbolic_steps[j]["state"] != g["state"]:
            s = symbolic_steps[j]
            out.append({"skill": s["skill"], "state": s["state"]})
            j += 1
        if j < len(symbolic_steps) and symbolic_steps[j]["state"] == g["state"]:
            out.append({"skill": g["skill"], "state": g["state"]})
            j += 1
    while j < len(symbolic_steps):
        s = symbolic_steps[j]
        out.append({"skill": s["skill"], "state": s["state"]})
        j += 1
    return out


def _ensure_tee_goal_tail(subgoals: list[dict], problem_str: str) -> list[dict]:
    out = list(subgoals)
    goal_region = _goal_region_from_problem(problem_str)
    if not goal_region:
        return out
    goal_state = f"(object-at tee {goal_region})"
    if out and out[-1].get("state") == goal_state:
        return out
    seen = {s["state"] for s in out}
    for e in _push_tee_subgoals_only(problem_str):
        if e["state"] not in seen:
            out.append(e)
            seen.add(e["state"])
        if e["state"] == goal_state:
            break
    return out


def get_subgoals(
    domain_path: str,
    problem_str: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    offline: bool = False,
    use_llm_first: bool = False,
) -> list[dict]:
    """
    Default: Gemini logical subgoals → merge with full symbolic plan (sound order/skills where needed).
    Fallbacks: symbolic plan only → BFS.

    use_llm_first=True: Gemini only (+ optional push_tee tail), no pyperplan repair.
    SUBGOAL_PDDL_FIRST=1 or LLM_PLAN_PLANNER_FIRST: pyperplan first (old behavior).
    """
    if offline or os.environ.get("LLM_PLAN_OFFLINE"):
        return compute_subgoals(problem_str)

    def _env_on(name: str) -> bool:
        v = (os.environ.get(name) or "").strip().lower()
        return v in ("1", "true", "yes", "on")

    if _env_on("SUBGOAL_PDDL_FIRST") or _env_on("LLM_PLAN_PLANNER_FIRST"):
        plan = run_pddl_planner(domain_path, problem_str)
        if plan:
            return plan_to_subgoals(plan, problem_str)
        return compute_subgoals(problem_str)

    if use_llm_first:
        with open(domain_path, "r") as f:
            domain_text = f.read()
        config = _load_config()
        try:
            gemini = _call_gemini_subgoals(domain_text, problem_str, model, temperature, config)
        except Exception:
            gemini = []
        if not gemini:
            return compute_subgoals(problem_str)
        return _ensure_tee_goal_tail(gemini, problem_str)

    with open(domain_path, "r") as f:
        domain_text = f.read()
    config = _load_config()
    gemini: list[dict] = []
    try:
        gemini = _call_gemini_subgoals(domain_text, problem_str, model, temperature, config)
    except Exception:
        gemini = []

    if not gemini:
        plan = run_pddl_planner(domain_path, problem_str)
        if plan:
            return plan_to_subgoals(plan, problem_str)
        return compute_subgoals(problem_str)

    gemini = _ensure_tee_goal_tail(gemini, problem_str)

    plan = run_pddl_planner(domain_path, problem_str)
    if plan:
        sym = plan_to_subgoals(plan, problem_str)
        if sym:
            return _align_gemini_to_symbolic(gemini, sym)

    return gemini


if __name__ == "__main__":
    from env_subgoal_runner import main

    main()
