from __future__ import annotations

from typing import Any, Dict
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.push_t import WhiteTableSceneBuilder

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils


@register_env('ReachGoal', max_episode_steps=1000)
class ReachGoalEnv(BaseEnv):
    """
    Bare table-top env: panda_stick arm only, no objects on the table.
    Matches the PushT-v1 physical setup exactly (same robot, white table, robot
    placement at x=-0.615) but with nothing on the table.

    Goal: move the EE to a randomly sampled 3-D position above the table.

    Observation (state mode, flat vector):
        qpos       (7,)   — 7 arm joints (panda_stick has no fingers)
        qvel       (7,)   — joint velocities
        ee_pos     (3,)   — EE world position
        goal_pos   (3,)   — goal world position
        ee_to_goal (3,)   — (goal - ee), vector the arm should close
    Total: 23-dim observation

    Action: pd_ee_delta_pose — 6D end-effector delta (Δpos + Δrot)
    Success criteria: ee < 2cm from goal
    """
    SUPPORTED_ROBOTS = ['panda_stick']

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_stick",
        robot_init_qpos_noise: float = 0.5,
        success_threshold: float = 0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.success_threshold = success_threshold
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        # Set the build-time pose to match where the robot will sit each episode,
        # suppressing the "no initial pose" warning and avoiding startup collisions.
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):  # noqa: ARG002
        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Initialize table and robot base placement
            self.table_scene.initialize(env_idx)

            # Override with explicit random qpos AFTER table_scene sets its default
            rest_qpos = torch.tensor(
                [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                device=self.device
            )
            noise = (torch.rand((b, 7), device=self.device) * 2 - 1) * self.robot_init_qpos_noise
            random_qpos = rest_qpos.unsqueeze(0) + noise
            self.agent.robot.set_qpos(random_qpos)
            self.agent.robot.set_qvel(torch.zeros((b, 7), device=self.device))

            # Sample a random goal in the arm's reachable workspace above the table
            goal_xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * 0.20  # ±20 cm
            goal_z  = torch.rand((b, 1), device=self.device) * 0.30 + 0.15     # 15–45 cm above table

            # Keep full-batch (N, 3) buffer so evaluate() and _get_obs_extra()
            # always see shape (N, 3) even during partial resets where only b < N envs reset
            if not hasattr(self, 'goal_pos') or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = torch.cat([goal_xy, goal_z], dim=1)

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        ee_pos  = self.agent.tcp.pose.p                         # (N, 3)
        dist    = torch.norm(ee_pos - self.goal_pos, dim=1)    # (N,)
        success = dist < self.success_threshold
        return {"success": success, "dist_to_goal": dist}

    def _get_obs_extra(self, info: Dict) -> Dict:
        ee_pos     = self.agent.tcp.pose.p          # (N, 3)
        ee_to_goal = self.goal_pos - ee_pos         # (N, 3)
        return {
            "goal_pos":   self.goal_pos,
            "ee_pos":     ee_pos,
            "ee_to_goal": ee_to_goal,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        dist          = info["dist_to_goal"]                    # (N,)
        reach_reward  = 1.0 - torch.tanh(5.0 * dist)
        success_bonus = info["success"].float() * 5.0
        return reach_reward + success_bonus

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Max possible reward ≈ 6.0 (tanh term ≈ 1 + bonus 5)
        return self.compute_dense_reward(obs, action, info) / 6.0
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)
