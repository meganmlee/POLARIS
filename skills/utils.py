"""
Shared skill constants, success-check helpers, and geometry utilities.

Numpy helpers are pure-numpy (no torch dependency).
Torch helpers (circle_overlap_frac_torch) lazy-import torch so this module
can be imported in any context without requiring a GPU environment.
"""
from __future__ import annotations

import math

import numpy as np


# ===========================================================================
# Reach
# ===========================================================================

class ReachCriteria:
    """Success thresholds for the reach skill."""
    SUCCESS_DIST     = 0.02   # strict: EE within 2 cm — env evaluate + PPO step check
    MPC_SUCCESS_DIST = 0.05   # lenient: MPC execute and PPO post-loop fallback
    MIN_EE_Z         = 0.10   # minimum EE z when building a reach goal in the executor


# ===========================================================================
# Pick
# ===========================================================================

class PickCriteria:
    """Success thresholds for the pick skill."""
    LIFT_THRESHOLD = 0.06     # cube centre z (m) must exceed this to count as lifted


class PickMPCStaging:
    """Phase-transition distances for the pick MPC state machine."""
    APPROACH_HEIGHT   = 0.06  # hover this far above cube before descending
    XY_ALIGN          = 0.015 # XY alignment threshold to start descending
    DESCEND_Z_TRIGGER = 0.02  # EE-to-cube Z gap before closing gripper
    LIFT_Z            = 0.12  # target EE z during lift phase


# ===========================================================================
# Place
# ===========================================================================

class PlaceCriteria:
    """Success thresholds for the place skill.

    Training and inference thresholds are intentionally different: the training
    env (PlaceSkillEnv) uses stricter values so the policy learns precision;
    the execute() functions use more lenient values to account for domain gap.
    """
    # Training env (PlaceSkillEnv.evaluate)
    TRAIN_PLACE_THRESH = 0.025  # cube-to-goal XY distance (m)
    TRAIN_RETREAT_DIST = 0.10   # EE-to-cube distance after release (m)

    # Inference execute() functions
    EXEC_PLACE_THRESH  = 0.05   # cube-to-goal XY distance (m)
    EXEC_RETREAT_DIST  = 0.05   # EE-to-cube distance after release (m)
    REST_Z_THRESH      = 0.05   # cube centre z < this → resting on table (m)
    GOAL_PLACE_Z       = 0.02   # z-coordinate of place target (cube centre on table)


class PlaceMPCStaging:
    """Phase-transition heights for the place MPC state machine."""
    HOVER_HEIGHT   = 0.08  # hover above goal before lowering (m)
    PLACE_HEIGHT   = 0.04  # target EE z when placing (m)
    RETREAT_HEIGHT = 0.20  # EE z to retreat to after release (m)
    CARRY_XY_TOL   = 0.02  # XY error (m) to transition from carry to lower
    LOWER_Z_SLACK  = 0.02  # EE z margin above PLACE_HEIGHT to trigger lower→release


# ===========================================================================
# PushCube (cube push)
# ===========================================================================

class PushCubeCriteria:
    """Success thresholds for the cube-push skill."""
    GOAL_THRESHOLD  = 0.05   # cube-to-goal XY distance (m)
    CONTACT_RADIUS  = 0.04   # EE-to-cube distance to consider in contact (m)


class PushCubeMPCStaging:
    """Phase-transition thresholds for the push-cube MPC state machine."""
    STAGE_OFFSET     = 0.05   # m behind cube centre along -push_dir
    STAGE_HIGH_Z     = 0.15   # travel height while repositioning in XY (m)
    PUSH_Z           = 0.03   # push height ≈ cube half-size (m)
    STAGE_XY_ALIGN   = 0.015  # XY error (m) to transition from hover to descend
    STAGE_ALIGN_DIST = 0.02   # total error (m) to transition from descend to push


# ===========================================================================
# PushO (disk push)
# ===========================================================================

class PushOCriteria:
    """Success thresholds and geometry for the disk-push skill."""
    DISK_RADIUS     = 0.05   # default disk radius (m) — must match PushOEnv
    SUCCESS_OVERLAP = 0.90   # minimum fractional circle overlap to count as success


class PushOMPCStaging:
    """Phase-transition thresholds for the push-O MPC state machine."""
    STAGE_OFFSET     = 0.08   # m behind disk centre along -push_dir
    STAGE_HIGH_Z     = 0.15   # travel height while repositioning in XY (m)
    PUSH_Z           = 0.02   # push height = disk half-thickness (m)
    STAGE_XY_ALIGN   = 0.015  # XY error (m) to transition from hover to descend
    STAGE_ALIGN_DIST = 0.02   # total error (m) to transition from descend to push


# ===========================================================================
# Geometry helpers — numpy (scalar inputs)
# ===========================================================================

def circle_overlap_frac(disk_xy: np.ndarray, goal_xy: np.ndarray, r: float) -> float:
    """Fraction of area that two equal circles of radius r share, given centre distance."""
    d = float(np.linalg.norm(disk_xy - goal_xy))
    if d >= 2.0 * r:
        return 0.0
    cos_arg = np.clip(d / (2.0 * r), -1.0 + 1e-7, 1.0 - 1e-7)
    A = 2.0 * r * r * np.arccos(cos_arg) - 0.5 * d * np.sqrt(max(4.0 * r * r - d * d, 0.0))
    return float(np.clip(A / (np.pi * r * r), 0.0, 1.0))


def circle_overlap_frac_torch(disk_xy, goal_xy, r: float):
    """Batched torch version of circle_overlap_frac for use in env.evaluate()."""
    import torch
    d = torch.norm(disk_xy - goal_xy, dim=-1).clamp(min=0.0)
    d_safe  = d.clamp(max=2.0 * r - 1e-6)
    cos_arg = (d_safe / (2.0 * r)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    A = (
        2.0 * r * r * torch.acos(cos_arg)
        - 0.5 * d_safe * torch.sqrt((4.0 * r * r - d_safe * d_safe).clamp(min=0.0))
    )
    frac = (A / (math.pi * r * r)).clamp(0.0, 1.0)
    return torch.where(d >= 2.0 * r, torch.zeros_like(frac), frac)


# ===========================================================================
# Success-check helpers — numpy (scalar inputs)
# ===========================================================================

def check_reach_success(ee_pos: np.ndarray, goal_xyz: np.ndarray) -> bool:
    return bool(np.linalg.norm(ee_pos - goal_xyz) < ReachCriteria.SUCCESS_DIST)


def check_pick_success(is_grasped: bool, cube_z: float) -> bool:
    return bool(is_grasped and cube_z > PickCriteria.LIFT_THRESHOLD)


def check_place_success(
    is_grasped: bool,
    cube_pos: np.ndarray,
    goal_xyz: np.ndarray,
    ee_pos: np.ndarray,
) -> bool:
    """Inference-time place success using EXEC_ thresholds."""
    cube_near = float(np.linalg.norm(cube_pos[:2] - goal_xyz[:2])) < PlaceCriteria.EXEC_PLACE_THRESH
    cube_rest = cube_pos[2] < PlaceCriteria.REST_Z_THRESH
    ee_away   = float(np.linalg.norm(ee_pos - cube_pos)) > PlaceCriteria.EXEC_RETREAT_DIST
    return bool((not is_grasped) and cube_near and cube_rest and ee_away)


def check_push_o_success(disk_xy: np.ndarray, goal_xy: np.ndarray, r: float) -> bool:
    return circle_overlap_frac(disk_xy, goal_xy, r) >= PushOCriteria.SUCCESS_OVERLAP


def check_push_cube_success(cube_xy: np.ndarray, goal_xy: np.ndarray) -> bool:
    return bool(np.linalg.norm(cube_xy - goal_xy) < PushCubeCriteria.GOAL_THRESHOLD)
