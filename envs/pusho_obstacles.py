"""PushO with obstacle blocks on the table, plus skill-specific subclasses."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import sapien
import torch
import mani_skill.envs  # noqa: F401
from mani_skill.agents.robots import Panda
from mani_skill.sensors.camera import CameraConfig
from envs.push_o_env import PushOEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


@register_env("PushO-WithObstacles-v1", max_episode_steps=200)
class PushOWithObstaclesEnv(PushOEnv):
    """PushO with multiple obstacle cubes of different sizes on the table."""

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # (half_size, rgba) for each obstacle — up to 10 cubes
    OBSTACLE_SPECS = [
        (0.020, [0.2, 0.6, 0.2, 1.0]),
        (0.015, [0.2, 0.4, 0.8, 1.0]),
        (0.025, [0.8, 0.4, 0.2, 1.0]),
        (0.018, [0.6, 0.2, 0.6, 1.0]),
        (0.022, [0.9, 0.8, 0.1, 1.0]),
        (0.017, [0.1, 0.8, 0.8, 1.0]),
        (0.020, [0.8, 0.2, 0.2, 1.0]),
        (0.015, [0.5, 0.5, 0.9, 1.0]),
        (0.023, [0.9, 0.5, 0.1, 1.0]),
        (0.018, [0.4, 0.9, 0.4, 1.0]),
    ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0, 0.9], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        self.obstacles = []
        for i, (half_size, color) in enumerate(self.OBSTACLE_SPECS):
            cube = actors.build_cube(
                self.scene,
                half_size=half_size,
                color=np.array(color, dtype=np.float32),
                name=f"obstacle_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.0, 0.0, half_size]),
            )
            self.obstacles.append(cube)

    # How many obstacles to place each episode: uniformly sampled from this range (inclusive).
    MIN_OBSTACLES: int = 5
    MAX_OBSTACLES: int = 10
    # Table half-extent (must stay within this XY radius of origin)
    TABLE_HALF: float = 0.25
    # Minimum distance between an obstacle and the disk centre
    MIN_DIST_FROM_DISK: float = 0.06
    # Minimum distance between two obstacles
    MIN_DIST_BETWEEN: float = 0.05

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            disk_pos = self.disk.pose.p[env_idx]  # (b, 3)
            q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)

            # Randomly choose how many obstacles to place this episode (same count for all envs in batch)
            n_active = int(torch.randint(self.MIN_OBSTACLES, self.MAX_OBSTACLES + 1, (1,)).item())

            placed_xy: list[torch.Tensor] = []  # (b, 2) tensors for collision checking

            for i, cube in enumerate(self.obstacles):
                half = self.OBSTACLE_SPECS[i][0]

                if i < n_active:
                    # Sample random XY positions with rejection for each env independently
                    xy = torch.zeros((b, 2), device=self.device)
                    for env_i in range(b):
                        disk_xy = disk_pos[env_i, :2]
                        for _ in range(50):
                            candidate = (torch.rand(2, device=self.device) * 2 - 1) * self.TABLE_HALF
                            # Reject if too close to disk
                            if torch.norm(candidate - disk_xy) < self.MIN_DIST_FROM_DISK:
                                continue
                            # Reject if too close to any already-placed obstacle
                            too_close = any(
                                torch.norm(candidate - p[env_i]) < self.MIN_DIST_BETWEEN
                                for p in placed_xy
                            )
                            if too_close:
                                continue
                            xy[env_i] = candidate
                            break
                        else:
                            # Fallback: fixed offset if sampling fails
                            fallback = [(0.08, 0.10), (-0.06, 0.12), (0.10, -0.08), (-0.10, -0.06)]
                            dx, dy = fallback[i % len(fallback)]
                            xy[env_i] = disk_xy + torch.tensor([dx, dy], device=self.device)

                    placed_xy.append(xy)
                    xyz = torch.cat([xy, torch.full((b, 1), half, device=self.device)], dim=1)
                else:
                    # Park inactive obstacles off-table so they don't interfere
                    park_x = 0.5 + i * 0.05
                    xyz = torch.tensor([[park_x, 0.0, half]], device=self.device).expand(b, 3)

                pose = Pose.create_from_pq(p=xyz, q=q_id.expand(b, 4))
                cube.set_pose(pose)


@register_env("Reach-WithObstacles-v1", max_episode_steps=200)
class ReachWithObstaclesEnv(PushOWithObstaclesEnv):
    """Move EE to a random goal above the table. Same scene as PushO-WithObstacles.

    Observation extras: goal_pos (3,), ee_pos (3,), ee_to_goal (3,)
    Success: EE within 2 cm of goal.
    """

    def __init__(self, *args, success_threshold: float = 0.02, **kwargs):
        self.success_threshold = success_threshold
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            goal_xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * 0.20
            goal_z  = torch.rand((b, 1), device=self.device) * 0.30 + 0.15
            if not hasattr(self, "goal_pos") or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = torch.cat([goal_xy, goal_z], dim=1)

    def evaluate(self, **kwargs):
        ee_pos  = self.agent.tcp.pose.p
        dist    = torch.norm(ee_pos - self.goal_pos, dim=1)
        return {"success": dist < self.success_threshold, "dist_to_goal": dist}

    def _get_obs_extra(self, info):
        ee_pos = self.agent.tcp.pose.p
        return {
            "goal_pos":   self.goal_pos,
            "ee_pos":     ee_pos,
            "ee_to_goal": self.goal_pos - ee_pos,
        }

    def compute_dense_reward(self, obs, action, info):
        reach_reward  = 1.0 - torch.tanh(5.0 * info["dist_to_goal"])
        success_bonus = info["success"].float() * 5.0
        return reach_reward + success_bonus

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.0


@register_env("PushCube-WithObstacles-v1", max_episode_steps=200)
class PushCubeWithObstaclesEnv(PushOWithObstaclesEnv):
    """Push a randomly selected obstacle cube to a goal position on the table.

    Each episode a different obstacle is chosen as the push target.
    Observation extras: ee_pos (3,), goal_cube_pos (3,), goal_pos (3,),
                        ee_to_goal_cube (3,), goal_cube_to_goal (3,)
    Success: goal cube XY within 5 cm of goal XY.
    """

    GOAL_THRESHOLD = 0.05

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # Cyan sphere: marks the goal cube's starting position each episode
        self.goal_cube_start_site = actors.build_sphere(
            self.scene,
            radius=self.GOAL_THRESHOLD,
            color=[0, 0.8, 1, 0.5],
            name="goal_cube_start_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_cube_start_site)
        # Green sphere: marks the goal position the cube must be pushed to
        self.goal_cube_target_site = actors.build_sphere(
            self.scene,
            radius=self.GOAL_THRESHOLD,
            color=[0, 1, 0, 0.5],
            name="goal_cube_target_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_cube_target_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)

            # Park the disk and goal_site off-table — they are unused in this env.
            q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, 4)
            park_p = torch.tensor([[2.0, 0.0, 0.05]], device=self.device).expand(b, 3)
            park_pose = Pose.create_from_pq(p=park_p, q=q_id)
            self.disk.set_pose(park_pose)
            self.goal_site.set_pose(park_pose)

            # Randomly pick which obstacle is the push target for each env
            if not hasattr(self, "goal_obstacle_idx") or self.goal_obstacle_idx.shape[0] != self.num_envs:
                self.goal_obstacle_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.goal_obstacle_idx[env_idx] = torch.randint(
                len(self.obstacles), (b,), device=self.device)

            # Sample goal XY across the full table (matches the planner grid ±0.30 m)
            goal_xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * self.TABLE_HALF
            if not hasattr(self, "goal_pos") or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = torch.cat(
                [goal_xy, torch.zeros((b, 1), device=self.device)], dim=1)

            # Place cyan start-position marker at the goal cube's initial XY
            all_obs_pos = torch.stack(
                [obs.pose.p for obs in self.obstacles], dim=0
            )  # (num_obstacles, num_envs, 3)
            goal_idx_b = self.goal_obstacle_idx[env_idx]  # (b,)
            start_xyz = all_obs_pos[goal_idx_b, env_idx]  # (b, 3)
            self.goal_cube_start_site.set_pose(Pose.create_from_pq(p=start_xyz, q=q_id))

            # Place green target marker at the sampled goal XY on the table surface
            target_xyz = torch.cat([goal_xy, torch.zeros((b, 1), device=self.device)], dim=1)
            self.goal_cube_target_site.set_pose(Pose.create_from_pq(p=target_xyz, q=q_id))

    def _get_goal_cube_pos(self) -> torch.Tensor:
        """Position of the selected goal obstacle for each env. Shape: (num_envs, 3)."""
        # (num_obstacles, num_envs, 3) → index per env
        all_pos = torch.stack([obs.pose.p for obs in self.obstacles], dim=0)
        return all_pos[self.goal_obstacle_idx, torch.arange(self.num_envs, device=self.device)]

    def evaluate(self, **kwargs):
        goal_cube_pos = self._get_goal_cube_pos()
        dist = torch.norm(goal_cube_pos[:, :2] - self.goal_pos[:, :2], dim=1)
        return {"success": dist < self.GOAL_THRESHOLD, "dist_cube_to_goal": dist}

    def _get_obs_extra(self, info):
        ee_pos        = self.agent.tcp.pose.p
        goal_cube_pos = self._get_goal_cube_pos()
        return {
            "ee_pos":            ee_pos,
            "goal_cube_pos":     goal_cube_pos,
            "goal_pos":          self.goal_pos,
            "ee_to_goal_cube":   goal_cube_pos - ee_pos,
            "goal_cube_to_goal": self.goal_pos - goal_cube_pos,
        }

    def compute_dense_reward(self, obs, action, info):
        ee_pos        = self.agent.tcp.pose.p
        goal_cube_pos = self._get_goal_cube_pos()
        reach_reward  = 1.0 - torch.tanh(5.0 * torch.norm(ee_pos - goal_cube_pos, dim=1))
        push_reward   = 1.0 - torch.tanh(5.0 * info["dist_cube_to_goal"])
        success_bonus = info["success"].float() * 5.0
        return reach_reward + push_reward + success_bonus

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 7.0


@register_env("PushO-WallObstacles-v1", max_episode_steps=200)
class PushOWallObstaclesEnv(PushOWithObstaclesEnv):
    """PushO where all 10 obstacle cubes are fixed in a wall across the table.

    All cubes are placed along WALL_ROW (grid row index), one per grid column.
    This guarantees every BFS path from disk to goal that crosses the wall is
    blocked, forcing the high-level planner to emit pick/place or push_cube
    subgoals rather than just routing around obstacles.

    Disk and goal positions are inherited (random from PushOEnv). Seeds where
    both land on the same side of the wall still work — the planner simply
    won't need to clear the wall for those episodes.
    """

    # Grid row for the wall. Row 4 → y ≈ -0.03 m (just below table centre).
    WALL_ROW: int = 4

    # Always use all 10 obstacles so every grid column is filled.
    MIN_OBSTACLES: int = 10
    MAX_OBSTACLES: int = 10

    # Planner constants (must match llm_plan.py)
    _TABLE_BOUND: float = 0.30
    _GRID: int = 10

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Let PushOWithObstaclesEnv set up disk, goal, robot positions first,
        # then immediately reposition every obstacle into the wall.
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)

            y_center = (self.WALL_ROW + 0.5) / self._GRID * 2 * self._TABLE_BOUND - self._TABLE_BOUND

            for col, cube in enumerate(self.obstacles):
                half = self.OBSTACLE_SPECS[col][0]
                x_center = (col + 0.5) / self._GRID * 2 * self._TABLE_BOUND - self._TABLE_BOUND
                # Clamp to physical table extent so cubes stay on the surface.
                x_center = float(np.clip(x_center, -self.TABLE_HALF, self.TABLE_HALF))
                xyz = torch.tensor(
                    [[x_center, y_center, half]], device=self.device
                ).expand(b, 3)
                pose = Pose.create_from_pq(p=xyz, q=q_id.expand(b, 4))
                cube.set_pose(pose)


@register_env("PickSkillEnv", max_episode_steps=100)
class PickSkillEnv(PushOWithObstaclesEnv):
    """Pick a randomly selected obstacle cube off the table.

    Each episode one of the inherited obstacle cubes is chosen as the pick
    target. The skill must grasp and lift it above `lift_threshold`.

    Observation extras: ee_pos (3,), pick_cube_pos (3,), ee_to_pick_cube (3,),
                        is_grasped (1,)
    Success: is_grasped AND pick_cube_pos z > lift_threshold.
    """

    # Absolute z height (cube centre) that counts as lifted off the table.
    # Obstacle half-sizes range 0.015–0.025, so 0.06 requires ~3–4 cm of lift.
    lift_threshold = 0.06

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            if not hasattr(self, "pick_obstacle_idx") or self.pick_obstacle_idx.shape[0] != self.num_envs:
                self.pick_obstacle_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.pick_obstacle_idx[env_idx] = torch.randint(
                len(self.obstacles), (b,), device=self.device)

    def _get_pick_cube_pos(self) -> torch.Tensor:
        """Position of the selected pick obstacle for each env. Shape: (num_envs, 3)."""
        all_pos = torch.stack([obs.pose.p for obs in self.obstacles], dim=0)
        return all_pos[self.pick_obstacle_idx, torch.arange(self.num_envs, device=self.device)]

    def evaluate(self) -> Dict[str, Any]:
        pick_cube_pos = self._get_pick_cube_pos()
        # Stack per-obstacle grasping flags then select the target per env
        is_grasped = torch.stack(
            [self.agent.is_grasping(obs) for obs in self.obstacles], dim=0
        )[self.pick_obstacle_idx, torch.arange(self.num_envs, device=self.device)]
        is_lifted = pick_cube_pos[:, 2] > self.lift_threshold
        return {
            "success": is_grasped & is_lifted,
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
        }

    def _get_obs_extra(self, info: Dict) -> Dict:
        ee_pos        = self.agent.tcp.pose.p
        pick_cube_pos = self._get_pick_cube_pos()
        return {
            "ee_pos":          ee_pos,
            "pick_cube_pos":   pick_cube_pos,
            "ee_to_pick_cube": pick_cube_pos - ee_pos,
            "is_grasped":      info["is_grasped"],
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        ee_pos        = self.agent.tcp.pose.p
        pick_cube_pos = self._get_pick_cube_pos()
        dist          = torch.linalg.norm(pick_cube_pos - ee_pos, dim=1)
        reach_reward  = 1.0 - torch.tanh(5.0 * dist)
        is_grasped    = info["is_grasped"].float()
        lift_reward   = 1.0 - torch.tanh(10.0 * (self.lift_threshold - pick_cube_pos[:, 2]).clamp(min=0.0))

        # Reward opening the gripper while the EE is near the cube, so the
        # policy learns to arrive with an open hand rather than ramming in closed.
        finger_pos  = self.agent.robot.get_qpos()[:, -2:]   # (N, 2) — last two joints are fingers
        gripper_openness = finger_pos.sum(dim=1) / (2 * 0.04)  # 0=closed, 1=fully open
        near_cube   = (dist < 0.06).float()
        pregrasp_reward = near_cube * gripper_openness

        reward = reach_reward + pregrasp_reward + is_grasped + lift_reward * is_grasped
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)
