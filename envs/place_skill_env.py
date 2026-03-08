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


@register_env("PlaceSkillEnv", max_episode_steps=100)
class PlaceSkillEnv(BaseEnv):
    """
    Place skill: lower and release a held cube onto a goal position on the table,
    then retreat the EE away.

    The robot starts every episode with the cube already firmly grasped in the
    gripper. Small joint noise is added to the arm configuration to prevent
    overfitting to a single holding pose. A goal position is sampled uniformly
    on the table surface and indicated by a green sphere marker.

    Initialization:
        - Table scene reset to default robot home pose.
        - Small noise (±qpos_noise rad) applied to each arm joint.
        - Fingers closed to cube_half_size - finger_grip_margin, creating slight
          interpenetration that physics resolves into a stable grasp.
        - Cube spawned at the TCP position after FK update.
        - Goal sampled uniformly within ±goal_spawn_half_size on the table.

    Success (all must hold simultaneously):
        - Cube is not grasped (fingers have released it).
        - Cube centre is within place_thresh of the goal position.
        - Cube is resting on the table (z ≈ cube_half_size).
        - EE has retreated > retreat_dist from the cube centre.

    Observation (state mode):
        tcp_pose        (7,)  EE world pose (pos + quat)
        obj_pose        (7,)  cube world pose
        goal_pos        (3,)  target position on table
        tcp_to_obj_pos  (3,)  vector from EE to cube centre
        obj_to_goal_pos (3,)  vector from cube to goal
        is_grasped      (1,)  whether cube is currently grasped
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    cube_half_size = 0.02

    # Arm joint noise at init (rad) — varies the holding pose to prevent overfitting
    qpos_noise = 0.05  # ±3°

    # Finger closure: slightly tighter than cube so physics resolves a firm grasp
    finger_grip_margin = 0.003  # fingers set to cube_half_size - margin

    # Goal sampled within this xy radius on the table
    goal_spawn_half_size = 0.10  # ±10 cm

    # Cube-to-goal distance for a successful place
    place_thresh = 0.025  # 2.5 cm

    # EE must retreat at least this far from cube centre to count as "away"
    retreat_dist = 0.10  # 10 cm

    # Cube is considered resting when its centre is at most this high above table
    rest_z_thresh = cube_half_size + 0.01  # 1 cm above nominal resting height

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
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.place_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # 1. Reset table scene to default home configuration
            self.table_scene.initialize(env_idx)

            # 2. Apply small noise to arm joints; close fingers firmly on cube.
            #    Panda DOF: 9 = 7 arm + 2 finger.
            #    Finger qpos = cube_half_size - margin creates slight interpenetration
            #    that the physics engine resolves into a stable contact grasp.
            current_qpos = self.agent.robot.get_qpos()[env_idx]  # (b, 9)
            arm_noise = (torch.rand((b, 7)) * 2 - 1) * self.qpos_noise
            finger_val = self.cube_half_size - self.finger_grip_margin
            finger_qpos = torch.full((b, 2), finger_val)
            new_qpos = torch.cat([current_qpos[:, :7] + arm_noise, finger_qpos], dim=1)
            self.agent.robot.set_qpos(new_qpos)
            self.agent.robot.set_qvel(torch.zeros_like(new_qpos))

            # 3. Spawn cube at the TCP position — FK is updated after set_qpos.
            #    A random yaw is applied so the cube orientation varies slightly.
            tcp_pos = self.agent.tcp_pose.p[env_idx]  # (b, 3)
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(tcp_pos, qs))

            # 4. Sample a random goal position on the table surface.
            goal_xy = (torch.rand((b, 2)) * 2 - 1) * self.goal_spawn_half_size
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = goal_xy
            goal_xyz[:, 2] = self.cube_half_size  # resting height on table
            if not hasattr(self, "goal_pos") or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = goal_xyz
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def evaluate(self) -> Dict[str, Any]:
        is_grasped = self.agent.is_grasping(self.cube)
        cube_pos = self.cube.pose.p  # (N, 3)
        tcp_pos = self.agent.tcp_pose.p  # (N, 3)

        obj_to_goal_dist = torch.linalg.norm(cube_pos - self.goal_pos, dim=1)

        # Cube must be released, near goal xy, and sitting flat on the table
        is_placed = (
            (~is_grasped)
            & (obj_to_goal_dist < self.place_thresh)
            & (cube_pos[:, 2] < self.rest_z_thresh)
        )

        # EE must have retreated from the cube
        tcp_to_obj_dist = torch.linalg.norm(tcp_pos - cube_pos, dim=1)
        is_retreated = tcp_to_obj_dist > self.retreat_dist

        return {
            "success": is_placed & is_retreated,
            "is_grasped": is_grasped,
            "is_placed": is_placed,
            "is_retreated": is_retreated,
            "obj_to_goal_dist": obj_to_goal_dist,
        }

    def _get_obs_extra(self, info: Dict) -> Dict:
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_pos,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_pos - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        is_grasped = info["is_grasped"].float()

        # Reward 1: move cube toward goal while grasped
        obj_to_goal_dist = info["obj_to_goal_dist"]
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward = place_reward * is_grasped

        # Reward 2: once placed, encourage EE to retreat
        tcp_to_obj_dist = torch.linalg.norm(
            self.agent.tcp_pose.p - self.cube.pose.p, dim=1
        )
        retreat_reward = 1 - torch.tanh(
            5 * (self.retreat_dist - tcp_to_obj_dist).clamp(min=0.0)
        )
        reward += retreat_reward * info["is_placed"].float()

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
