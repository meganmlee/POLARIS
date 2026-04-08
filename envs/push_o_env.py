"""PushO-v1: Push a circular disk to a matching goal region on the table."""
from __future__ import annotations

from typing import Any, Dict

import sapien
import sapien.physx as physx
import sapien.render
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# SAPIEN 3 cylinders have their principal axis along local-x.
# Rotating -90° around y maps x→z so the disk lies flat on the table.
_DISK_QUAT = [0.7071, 0.0, -0.7071, 0.0]


@register_env("PushO-v1", max_episode_steps=200)
class PushOEnv(BaseEnv):
    """Push a circular disk (flat cylinder) to a same-radius goal region on the table.

    Success when the disk overlaps ≥90% with the goal circle, computed analytically
    from the XY distance between their centres.

    Observation extras: ee_pos(3), disk_pos(3), goal_pos(3),
                        ee_to_disk(3), disk_to_goal(3), overlap_frac(1)
    Reward: reach + push + overlap + finger_close; terminal bonus 5.0 on success.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    disk_radius: float = 0.05
    disk_half_thickness: float = 0.02
    disk_static_friction: float = 0.3
    disk_dynamic_friction: float = 0.3

    GOAL_HALF: float = 0.15
    DISK_SPAWN_HALF: float = 0.12
    MIN_DISK_GOAL_DIST: float = 0.08
    SUCCESS_OVERLAP: float = 0.90

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        disk_radius: float = 0.05,
        disk_half_thickness: float = 0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.disk_radius = disk_radius
        self.disk_half_thickness = disk_half_thickness
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        pm = physx.PhysxMaterial(
            static_friction=self.disk_static_friction,
            dynamic_friction=self.disk_dynamic_friction,
            restitution=0.1,
        )
        vm_disk = sapien.render.RenderMaterial(
            base_color=[0.85, 0.2, 0.2, 1.0], metallic=0.0, roughness=0.5
        )
        vm_goal = sapien.render.RenderMaterial(
            base_color=[0.2, 0.85, 0.2, 0.35], metallic=0.0, roughness=1.0
        )

        disk_builder = self.scene.create_actor_builder()
        disk_builder.add_cylinder_collision(
            radius=self.disk_radius, half_length=self.disk_half_thickness, material=pm
        )
        disk_builder.add_cylinder_visual(
            radius=self.disk_radius, half_length=self.disk_half_thickness, material=vm_disk
        )
        disk_builder.initial_pose = sapien.Pose(
            p=[0.0, 0.0, self.disk_half_thickness], q=_DISK_QUAT
        )
        self.disk = disk_builder.build(name="push_disk")

        goal_half_t = self.disk_half_thickness * 0.4
        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_cylinder_visual(
            radius=self.disk_radius, half_length=goal_half_t, material=vm_goal
        )
        goal_builder.initial_pose = sapien.Pose(
            p=[0.1, 0.0, goal_half_t], q=_DISK_QUAT
        )
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            q = torch.tensor(_DISK_QUAT, device=self.device).unsqueeze(0).expand(b, 4)

            goal_xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * self.GOAL_HALF
            goal_z  = torch.full((b, 1), self.disk_half_thickness * 0.4, device=self.device)
            if not hasattr(self, "goal_pos") or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = torch.cat([goal_xy, goal_z], dim=1)

            disk_xy = torch.zeros((b, 2), device=self.device)
            for i in range(b):
                for _ in range(50):
                    candidate = (torch.rand(2, device=self.device) * 2 - 1) * self.DISK_SPAWN_HALF
                    if torch.norm(candidate - goal_xy[i]) >= self.MIN_DISK_GOAL_DIST:
                        disk_xy[i] = candidate
                        break
                else:
                    disk_xy[i] = -goal_xy[i].clamp(-self.DISK_SPAWN_HALF, self.DISK_SPAWN_HALF)

            disk_z   = torch.full((b, 1), self.disk_half_thickness, device=self.device)
            disk_xyz = torch.cat([disk_xy, disk_z], dim=1)

            self.disk.set_pose(Pose.create_from_pq(p=disk_xyz, q=q))
            self.goal_site.set_pose(Pose.create_from_pq(p=self.goal_pos[env_idx], q=q))

    def _circle_overlap_frac(self, disk_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        r = self.disk_radius
        d = torch.norm(disk_xy - goal_xy, dim=-1).clamp(min=0.0)
        d_safe  = d.clamp(max=2.0 * r - 1e-6)
        cos_arg = (d_safe / (2.0 * r)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        A = (
            2.0 * r * r * torch.acos(cos_arg)
            - 0.5 * d_safe * torch.sqrt((4.0 * r * r - d_safe * d_safe).clamp(min=0.0))
        )
        frac = (A / (torch.pi * r * r)).clamp(0.0, 1.0)
        return torch.where(d >= 2.0 * r, torch.zeros_like(frac), frac)

    def evaluate(self) -> Dict[str, Any]:
        disk_xy = self.disk.pose.p[:, :2]
        goal_xy = self.goal_pos[:, :2]
        overlap  = self._circle_overlap_frac(disk_xy, goal_xy)
        return {
            "success":      overlap >= self.SUCCESS_OVERLAP,
            "overlap_frac": overlap,
            "dist_to_goal": torch.norm(disk_xy - goal_xy, dim=1),
        }

    def _get_obs_extra(self, info: Dict) -> Dict:
        ee_pos   = self.agent.tcp.pose.p
        disk_pos = self.disk.pose.p
        return {
            "ee_pos":       ee_pos,
            "ee_to_disk":   disk_pos - ee_pos,
            "disk_to_goal": self.goal_pos - disk_pos,
            "overlap_frac": info["overlap_frac"].unsqueeze(-1),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        ee_pos   = self.agent.tcp.pose.p
        disk_pos = self.disk.pose.p

        reach_reward   = 0.5 * (1.0 - torch.tanh(5.0 * torch.norm(ee_pos - disk_pos, dim=1)))
        push_reward    = 1.0 - torch.tanh(5.0 * info["dist_to_goal"])
        overlap_reward = info["overlap_frac"]

        # Reward closing the fingers to form a solid pushing surface.
        # Panda finger joints each range [0, 0.04] m; sum=0 → fully closed.
        finger_qpos        = self.agent.robot.get_qpos()[:, -2:]
        gripper_closedness = 1.0 - finger_qpos.sum(dim=1) / (2 * 0.04)
        finger_reward      = 0#.3 * gripper_closedness

        reward = reach_reward + push_reward + overlap_reward + finger_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # max non-terminal = 3.5, terminal bonus = 5.0 → normalised ceiling = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.0, 0.8], target=[0.0, 0.0, 0.0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
