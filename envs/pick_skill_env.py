from __future__ import annotations

from typing import Any, Dict

import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PickSkillEnv", max_episode_steps=100)
class PickSkillEnv(BaseEnv):
    """
    Pick skill: close the gripper around a cube given the EE already starts near it.

    The robot (panda) is initialised at its default table-scene ready pose.
    The cube is spawned within a small radius of the EE's projected xy position,
    so no approach motion is required — the skill only needs to grasp and lift.

    Episode terminates successfully once the gripper is grasping the cube AND
    the cube has been lifted at least `lift_threshold` above the table surface.
    No further motion planning (placing, moving to goal) is part of this skill.

    Observation (state mode):
        tcp_pose       (7,)  EE world pose (pos + quat)
        obj_pose       (7,)  cube world pose
        tcp_to_obj_pos (3,)  vector from EE to cube centre
        is_grasped     (1,)  whether cube is currently grasped

    Success: is_grasped AND cube lifted > lift_threshold above table.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    cube_half_size = 0.02

    # Cube spawns within this xy radius of the EE's default position
    cube_spawn_noise = 0.03  # ±3 cm

    # Cube must be lifted this far above the table surface to count as picked
    lift_threshold = 0.015  # 1.5 cm

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Reset robot and table to default ready configuration
            self.table_scene.initialize(env_idx)

            # Spawn cube near the EE's xy projection so no approach is needed.
            # The panda default pose puts the TCP roughly above (0, 0); spawning
            # with small noise ensures the EE is already close at episode start.
            noise_xy = (torch.rand((b, 2)) * 2 - 1) * self.cube_spawn_noise
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = noise_xy
            xyz[:, 2] = self.cube_half_size  # resting on table (table top at z=0)

            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

    def evaluate(self) -> Dict[str, Any]:
        is_grasped = self.agent.is_grasping(self.cube)
        # Cube centre must be above table + half-size + lift margin
        is_lifted = self.cube.pose.p[:, 2] > self.cube_half_size + self.lift_threshold
        return {
            "success": is_grasped & is_lifted,
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
        }

    def _get_obs_extra(self, info: Dict) -> Dict:
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Reach: encourage EE to move toward cube
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, dim=1
        )
        reach_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reach_reward

        # Grasp: bonus when fingers close around cube
        is_grasped = info["is_grasped"].float()
        reward += is_grasped

        # Lift: encourage raising cube off the table, gated on grasp
        lift_dist = (self.cube.pose.p[:, 2] - self.cube_half_size).clamp(min=0.0)
        lift_reward = 1 - torch.tanh(10 * (self.lift_threshold - lift_dist).clamp(min=0.0))
        reward += lift_reward * is_grasped

        # Terminal bonus
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Max achievable reward ≈ 5 (terminal bonus)
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)
