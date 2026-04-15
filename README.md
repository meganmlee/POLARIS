# POLARIS

POLARIS (Policy Assignment for Robust Intent-Driven Skill Execution) is a robot manipulation framework that combines LLM-based task decomposition with an adaptive skill library. Long-horizon tasks are broken into subgoals by an LLM, which compresses the simulation state into a PDDL representation of object states and constraints. For each subgoal, a **Policy Assignment** module selects the most efficient execution method — classical motion planning (MPPI) or a learned RL policy (PPO) — and retries with reassignment on failure.

## System Components

### Adaptive Task Decomposition
- An LLM breaks long-horizon tasks into a sequence of primitive subgoals
- The simulation state is compressed into a PDDL-style representation of object states and constraints, which is passed as context to the LLM

### Policy Assignment
- For each subgoal, evaluates **classical trajectory optimizer** (MPPI, Cartesian) vs **learned RL policies** (PPO) on efficiency and success rate
- On failure, reassigns the subgoal to an alternative method from the skill library
- The skill library is designed to grow — new classical or learned backends can be added per skill

### Skills Library
Each skill has multiple backends sharing the same environment, so methods are directly comparable:

| Skill | Classical | Learned (PPO) | Environment |
|---|---|---|---|
| Reach | MPPI (`reach_mpc.py`) | PPO (`reach_ppo.py`) | `Reach-WithObstacles-v1` |
| Push Cube | MPPI (`push_cube_mpc.py`) | PPO (`push_cube_ppo.py`) | `PushCube-WithObstacles-v1` |
| Pick | MPPI (`pick_cube_mpc.py`) | PPO (`pick_cube_ppo.py`) | `PickSkillEnv` |
| Place | MPPI (`place_cube_mpc.py`) | PPO (`place_cube_ppo.py`) | `PlaceSkillEnv` |
| Push O | — | — | `PushO-WithObstacles-v1` |

## Project Structure

```
POLARIS/
├── envs/                          # Custom ManiSkill environments (shared across methods)
│   ├── pusho_obstacles.py         # PushO, PushCube, Reach with obstacle variants
│   ├── shelf_retrieve_v1.py       # Object retrieval from shelf
│   └── shelf_scene_builder.py
├── skills/                        # Skill backends
│   ├── mpc_base.py                # Shared MPPI infrastructure
│   ├── push_cube/
│   │   ├── push_cube_ppo.py       # Learned: PPO policy
│   │   └── push_cube_mpc.py       # Classical: MPPI controller
│   ├── pick/
│   │   ├── pick_cube_ppo.py       # Learned: PPO policy
│   │   └── pick_cube_mpc.py       # Classical: MPPI controller
│   ├── place/
│   │   ├── place_cube_ppo.py      # Learned: PPO policy
│   │   └── place_cube_mpc.py      # Classical: MPPI controller
│   └── reach/
│       ├── reach_ppo.py           # Learned: PPO policy
│       └── reach_mpc.py           # Classical: MPPI controller
├── examples/                      # Demos and visualization
│   ├── push_cube_ppo_demo.py
│   ├── pusho_obstacles_demo.py
│   ├── reach_demo.py              # Proportional controller baseline
│   ├── reach_ppo_demo.py
│   ├── reach_mpc_demo.py          # MPPI reach demo
│   ├── pick_cube_mpc_demo.py      # MPPI pick demo
│   └── place_cube_mpc_demo.py     # MPPI place demo
├── high_level_planner/            # LLM-based subgoal generation
│   ├── llm_plan.py                # PDDL problem builder + LLM/symbolic/BFS planner
│   ├── env_subgoal_runner.py      # Live env or dummy state → subgoals
│   ├── executor.py                # Closed-loop executor: plan → execute skill → re-plan
│   ├── domain_pusho.pddl          # PDDL domain (actions, predicates)
│   └── config.json.example        # Gemini API config template
├── planning_wrapper/              # State clone/restore utilities for planning backends
└── checkpoints/                          # Training checkpoints (auto-created)
```

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/meganmlee/POLARIS.git
cd POLARIS
```

### Step 2: Create Conda Environment

```bash
conda create -n polaris python=3.11 -y
conda activate polaris
```

### Step 3: Install Pinocchio

```bash
conda install -c conda-forge pinocchio -y
```

### Step 4: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Install ManiSkill3

```bash
pip install git+https://github.com/haosulab/ManiSkill.git
```

### Step 6: Install PyTorch

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio
```

### Step 7: Install the Package

```bash
pip install -e .
```

### Verify

```bash
python -c "import mani_skill; print('ManiSkill3 OK')"
python -c "import envs; print('Custom envs OK')"
python -c "import torch; print('PyTorch', torch.__version__)"
```

## Environments

All environments are in [envs/](envs/) and registered on `import envs`. RL and planning backends for each skill use the **same environment** so performance is directly comparable.

| Environment ID | Base Task | Notes |
|---|---|---|
| `Reach-WithObstacles-v1` | MoveToPoint | 4 randomized obstacle cubes |
| `PushCube-WithObstacles-v1` | PushCube | 4 randomized obstacle cubes |
| `PushO-WithObstacles-v1` | PushO | 4 randomized obstacle cubes |

## Training & Evaluation

### Learned Skills (PPO)

```bash
# Train
cd skills/<skill>
python reach_ppo.py
python push_cube_ppo.py

# With options
python push_cube_ppo.py \
    --num_envs 512 \
    --total_timesteps 10_000_000 \
    --seed 42

# Evaluate a checkpoint
python push_cube_ppo.py \
    --evaluate \
    --checkpoint checkpoints/PushCube-WithObstacles-v1__1__<timestamp>/final_ckpt.pt
```

### Classical Skills

No training required.

```bash
python skills/reach/reach_mpc.py
python skills/reach/reach_mpc.py --num_episodes 20 --seed 42
python skills/push_cube/push_cube_mpc.py
python skills/pick/pick_cube_mpc.py
python skills/place/place_cube_mpc.py
```

## Running Demos

```bash
# Reach
python examples/reach_demo.py        # proportional controller baseline
python examples/reach_ppo_demo.py --checkpoint checkpoints/<run>/final_ckpt.pt  # PPO policy
python examples/reach_mpc_demo.py    # MPPI controller

# Pick
python examples/pick_cube_mpc_demo.py    # MPPI pick controller

# Place
python examples/place_cube_mpc_demo.py   # MPPI place controller
python examples/reach_ppo_demo.py --checkpoint checkpoints/<run>/final_ckpt.pt --seed 10 # change seed randomization (default 5)

# PushCube
python examples/push_cube_ppo_demo.py
python examples/push_cube_ppo_demo.py --checkpoint checkpoints/<run>/final_ckpt.pt

# PushO
python examples/pusho_obstacles_demo.py
```

## Planning Wrapper

Planning backends (MPC previews, rollouts, etc.) require branching and backtracking without a full simulator reset. The `planning_wrapper` package provides state cloning and restoration for any ManiSkill3 environment.

```python
from planning_wrapper import ManiSkillPlanningWrapper
from planning_wrapper.adapters import PushOTaskAdapter
import gymnasium as gym
import envs

env = gym.make("PushO-WithObstacles-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose")
wrapper = ManiSkillPlanningWrapper(env, adapter=PushOTaskAdapter())

obs, info = wrapper.reset(seed=0)
snapshot = wrapper.clone_state()   # save branch point
# ... planner searches ...
wrapper.restore_state(snapshot)    # backtrack
```

Available adapters: `PushOTaskAdapter`, `ShelfRetrieveTaskAdapter`.

## High-Level Planner

The `high_level_planner/` module handles LLM-based subgoal generation and closed-loop execution.

### `llm_plan.py` — Subgoal Planner

Converts the current simulation state into a discretized grid representation (PDDL-style), then generates a subgoal sequence using one of three methods in priority order:

1. **Gemini** — queries an LLM for a natural-language plan, grounded to PDDL predicates
2. **pyperplan** — falls back to a symbolic PDDL planner if the LLM is unavailable or fails
3. **BFS** — last-resort graph search over the discretized state space

The LLM plan is validated and corrected against the symbolic planner output when both are available.

```bash
python high_level_planner/llm_plan.py              # LLM + symbolic fallback
python high_level_planner/llm_plan.py --offline    # BFS only (no API needed)
python high_level_planner/llm_plan.py --live       # read state from running simulator
python high_level_planner/llm_plan.py --live --llm # LLM only, skip symbolic planner
SUBGOAL_PDDL_FIRST=1 python high_level_planner/llm_plan.py --live  # symbolic planner first
```

Example output:
```
reach	(robot-at robot1 r_5_5)
push_disk	(object-at disk r_4_5)
push_disk	(object-at disk r_3_5)
```

### `env_subgoal_runner.py` — State Builder

Reads the current environment state (object, goal, robot hand, obstacles) and feeds it to `llm_plan.py`. Run with `--live` to use a real ManiSkill simulator, or without flags to use dummy state for offline testing.

### `executor.py` — Closed-Loop Executor

Ties the planner and skill library into a full execution loop: plan → dispatch skill → detect failure → re-plan.

**Supported skills:** `reach` (MPPI or PPO), `pick` (MPPI or PPO), `place` (MPPI or PPO), `push_cube` (MPPI or PPO)

```bash
# PPO reach skill (checkpoint required)
python high_level_planner/executor.py --seed 42 --reach-checkpoint Reach-WithObstacles-v1__1__<timestamp>

# MPPI reach skill (no checkpoint needed)
python high_level_planner/executor.py --seed 42 --skill mpc

# Offline with MPPI, limit replans
python high_level_planner/executor.py --seed 3 --max_replans 5 --offline --skill mpc

# Full multi-skill run
python high_level_planner/executor.py --seed 0 \
    --reach-checkpoint Reach-WithObstacles-v1__1__<ts> \
    --push-cube-checkpoint PushCube-WithObstacles-v1__1__<ts>

# Full multi-skill run with default checkpoints for all PPO skills
python high_level_planner/executor.py
```

Checkpoint args accept either a full path or a run name under `checkpoints/` (e.g. `Reach-WithObstacles-v1__1__1773025568` expands to `checkpoints/.../final_ckpt.pt`). By default, all checkpoints under `checkpoints/` with short names (e.g. PickSkill) without timestamps will be used.

### `domain_pusho.pddl` — PDDL Domain

Defines the world model used by the symbolic planner: types (`robot`, `disk`, `obstacle`, `region`), predicates (`robot-at`, `object-at`, `obstacle-at`, `holding`, `pickable`, `push-only`, `clear`, `adjacent`), and actions (`reach`, `push_disk`, `pick`, `place`, `push_cube`).

### `config.json.example` — Gemini Config Template

Copy to `config.json` and fill in your Vertex AI project and location, or set them as environment variables:

```bash
export VERTEX_PROJECT=your_project
export VERTEX_LOCATION=us-central1
```

## Requirements

- Python 3.11
- ManiSkill3
- PyTorch (GPU recommended for PPO training)
- Pinocchio (via conda)

See [requirements.txt](requirements.txt) for the full dependency list.

## Contributors
- Abhishek Mathur
- Megan Lee
- Tom Gao