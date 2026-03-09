"""PushT with obstacle blocks on the table, plus skill-specific subclasses."""
import numpy as np
import sapien
import torch
import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


@register_env("PushT-WithObstacles-v1", max_episode_steps=200)
class PushTWithObstaclesEnv(PushTEnv):
    """PushT with multiple obstacle cubes of different sizes on the table."""

    # (half_size, rgba) for each obstacle
    OBSTACLE_SPECS = [
        (0.02, [0.2, 0.6, 0.2, 1.0]),
        (0.015, [0.2, 0.4, 0.8, 1.0]),
        (0.025, [0.8, 0.4, 0.2, 1.0]),
        (0.018, [0.6, 0.2, 0.6, 1.0]),
    ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0, 0.9], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)

    def _load_scene(self, options: dict):
        self.T_mass = 0.3
        self.T_dynamic_friction = 0.8
        self.T_static_friction = 0.8
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            tee_pos = self.tee.pose.p[env_idx]  # (b, 3) — only the envs being reset
            # Place obstacles around the table (offsets from tee so they don't overlap T)
            offsets_xy = [
                (0.08, 0.10),
                (-0.06, 0.12),
                (0.10, -0.08),
                (-0.10, -0.06),
            ]
            for i, cube in enumerate(self.obstacles):
                half = self.OBSTACLE_SPECS[i][0]
                dx, dy = offsets_xy[i] if i < len(offsets_xy) else (0.05, 0.05)
                xyz = tee_pos.clone()
                xyz[:, 0] += dx
                xyz[:, 1] += dy
                xyz[:, 2] = half
                q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, 4)
                pose = Pose.create_from_pq(p=xyz, q=q)
                cube.set_pose(pose)


@register_env("MoveGoal-WithObstacles-v1", max_episode_steps=200)
class MoveGoalWithObstaclesEnv(PushTWithObstaclesEnv):
    """Move EE to a random goal above the table. Same scene as PushT-WithObstacles.

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
class PushCubeWithObstaclesEnv(PushTWithObstaclesEnv):
    """Push a randomly selected obstacle cube to a goal position on the table.

    Each episode a different obstacle is chosen as the push target.
    Observation extras: ee_pos (3,), goal_cube_pos (3,), goal_pos (3,),
                        ee_to_goal_cube (3,), goal_cube_to_goal (3,)
    Success: goal cube XY within 5 cm of goal XY.
    """

    GOAL_THRESHOLD = 0.05

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)

            # Randomly pick which obstacle is the push target for each env
            if not hasattr(self, "goal_obstacle_idx") or self.goal_obstacle_idx.shape[0] != self.num_envs:
                self.goal_obstacle_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.goal_obstacle_idx[env_idx] = torch.randint(
                len(self.obstacles), (b,), device=self.device)

            # Sample goal XY on table surface (stored as 3-D with z=0)
            goal_xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * 0.15
            if not hasattr(self, "goal_pos") or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = torch.cat(
                [goal_xy, torch.zeros((b, 1), device=self.device)], dim=1)

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
