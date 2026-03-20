"""
Skill: push a randomly placed cube to a random goal region on the table.
The built-in reward guides the arm to get behind the cube first, then push it.

Run training:
    python push_cube_ppo.py

Evaluate:
    python push_cube_ppo.py --evaluate --checkpoint runs/<run>/final_ckpt.pt
"""
from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import envs  # noqa: F401 — registers PushCube-WithObstacles-v1
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    cuda: bool = True

    env_id: str = "PushCube-WithObstacles-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    num_envs: int = 512
    num_eval_envs: int = 8
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    num_steps: int = 50
    num_eval_steps: int = 50
    partial_reset: bool = True
    eval_partial_reset: bool = False

    # PPO
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    finite_horizon_gae: bool = False

    eval_freq: int = 25
    save_model: bool = True
    capture_video: bool = True
    save_eval_video_freq: Optional[int] = 5  # save eval video every N eval runs; None = never

    evaluate: bool = False
    checkpoint: Optional[str] = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        # PushCube state obs includes robot qpos/qvel, EE pose, cube pose,
        # cube-to-goal vector — larger than reach, same MLP handles it fine
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)),     nn.Tanh(),
            layer_init(nn.Linear(256, 256)),     nn.Tanh(),
            layer_init(nn.Linear(256, 1),        std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)),     nn.Tanh(),
            layer_init(nn.Linear(256, 256)),     nn.Tanh(),
            layer_init(nn.Linear(256, act_dim),  std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        mean = self.actor_mean(x)
        if deterministic:
            return mean
        return Normal(mean, self.actor_logstd.expand_as(mean).exp()).sample()

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std  = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)


# ---------------------------------------------------------------------------
# Callable skill API
# ---------------------------------------------------------------------------

def _build_push_cube_obs(obs: dict, raw_env, obstacle, goal_xyz: np.ndarray) -> np.ndarray:
    """
    Reconstruct the flat state obs that PushCubeWithObstaclesEnv produces during training.
    Layout: qpos(9) + qvel(9) + ee_pos(3) + goal_cube_pos(3) + goal_pos(3)
            + ee_to_goal_cube(3) + goal_cube_to_goal(3) = 33
    """
    qpos     = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)
    qvel     = np.asarray(obs["agent"]["qvel"], dtype=np.float32).reshape(-1)
    ee_pos   = raw_env.agent.tcp.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    goal     = goal_xyz.astype(np.float32).reshape(-1)[:3]
    return np.concatenate([
        qpos, qvel, ee_pos, cube_pos, goal,
        cube_pos - ee_pos,   # ee_to_goal_cube
        goal - cube_pos,     # goal_cube_to_goal
    ])


def execute(
    env,
    obs: dict,
    block_idx: int,
    goal_xyz: np.ndarray,
    checkpoint: str,
    max_steps: int = 200,
    render: bool = False,
    device: str = "cpu",
) -> tuple[bool, dict]:
    """
    Run the PPO push-cube policy on an already-running PushT env to push
    obstacle[block_idx] to goal_xyz.
    Requires a checkpoint trained on PushCube-WithObstacles-v1 with obs_mode='state'.
    Returns (success, latest_obs).
    """
    import types
    raw      = env.unwrapped
    obstacle = raw.obstacles[block_idx]

    GOAL_THRESHOLD = 0.05

    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    obs_dim = state_dict["actor_mean.0.weight"].shape[1]
    act_dim = state_dict["actor_mean.6.weight"].shape[0]
    env_ns  = types.SimpleNamespace(
        single_observation_space=types.SimpleNamespace(shape=(obs_dim,)),
        single_action_space=types.SimpleNamespace(shape=(act_dim,)),
    )
    agent = Agent(env_ns).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    action_low  = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)

    current_obs = obs
    for _ in range(max_steps):
        flat  = _build_push_cube_obs(current_obs, raw, obstacle, goal_xyz)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)
        current_obs, _, term, trunc, _ = env.step(action)
        if render:
            env.render()
        cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        if float(np.linalg.norm(cube_pos[:2] - goal_xyz[:2])) < GOAL_THRESHOLD:
            return True, current_obs
        if np.asarray(term).any() or np.asarray(trunc).any():
            break

    cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    return bool(np.linalg.norm(cube_pos[:2] - goal_xyz[:2]) < GOAL_THRESHOLD), current_obs


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)

    args.batch_size     = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = args.exp_name or f"{args.env_id}__{args.seed}__{int(time.time())}"
    run_dir = str(Path(__file__).resolve().parents[2] / "checkpoints" / run_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(obs_mode=args.obs_mode, control_mode=args.control_mode,
                      render_mode="rgb_array", sim_backend="physx_cuda")

    envs = gym.make(args.env_id,
                    num_envs=args.num_envs if not args.evaluate else 1,
                    reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs      = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    _eval_run = [0]  # mutable counter used by the eval video trigger closure
    if args.capture_video:
        eval_output_dir = f"{run_dir}/eval_videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"

        _is_evaluate = args.evaluate
        _save_eval_video_freq = args.save_eval_video_freq
        def eval_video_trigger(_):
            if _is_evaluate:
                return True
            if _save_eval_video_freq is None:
                return False
            return _eval_run[0] % _save_eval_video_freq == 0
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir,
                                  save_trajectory=args.evaluate, trajectory_name="trajectory",
                                  max_steps_per_video=args.num_eval_steps, video_fps=30,
                                  save_video_trigger=eval_video_trigger)

    envs      = ManiSkillVectorEnv(envs,      args.num_envs,
                                   ignore_terminations=not args.partial_reset,      record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs,
                                   ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent     = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.evaluate:
        print("=== Evaluation mode ===")
        eval_obs, _ = eval_envs.reset()
        eval_metrics = defaultdict(list)
        for _ in range(args.num_eval_steps):
            with torch.no_grad():
                eval_obs, _, _, _, eval_infos = eval_envs.step(
                    agent.get_action(eval_obs, deterministic=True))
                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v[mask].float())
        for k, v in eval_metrics.items():
            print(f"  {k}: {torch.stack(v).float().mean():.4f}")
        envs.close(); eval_envs.close()
        exit()

    writer = SummaryWriter(run_dir)
    writer.add_text("hyperparameters", str(vars(args)))
    print(f"Training {args.env_id}  envs={args.num_envs}  batch={args.batch_size}  iters={args.num_iterations}")

    obs_buf  = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    act_buf  = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape,      device=device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf  = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf  = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done   = torch.zeros(args.num_envs, device=device)

    action_low  = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)
    clip_action = lambda a: torch.clamp(a.detach(), action_low, action_high)

    global_step = 0
    start_time  = time.time()

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iter {iteration}/{args.num_iterations}  step={global_step}")

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        agent.eval()
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        rollout_t = time.time()

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step]  = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            act_buf[step]  = action
            logp_buf[step] = logp
            val_buf[step]  = value.flatten()

            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).float()
            rew_buf[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    writer.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = \
                        agent.get_value(infos["final_observation"][done_mask]).view(-1)

        rollout_time = time.time() - rollout_t

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            adv_buf    = torch.zeros_like(rew_buf)
            lastgae    = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nnd = 1.0 - next_done
                    nv  = torch.zeros_like(next_value) if args.finite_horizon_gae else next_value
                else:
                    nnd, nv = 1.0 - done_buf[t + 1], val_buf[t + 1]
                real_nv = nnd * nv + final_values[t]
                delta      = rew_buf[t] + args.gamma * real_nv - val_buf[t]
                adv_buf[t] = lastgae = delta + args.gamma * args.gae_lambda * nnd * lastgae
            ret_buf = adv_buf + val_buf

        b_obs  = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_act  = act_buf.reshape((-1,) + envs.single_action_space.shape)
        b_logp = logp_buf.reshape(-1)
        b_adv  = adv_buf.reshape(-1)
        b_ret  = ret_buf.reshape(-1)
        b_val  = val_buf.reshape(-1)

        agent.train()
        inds = np.arange(args.batch_size)
        clipfracs = []
        update_t  = time.time()

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = inds[start: start + args.minibatch_size]
                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio    = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1).abs() > args.clip_coef).float().mean().item())

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_adv = b_adv[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                ).mean()
                vf_loss = 0.5 * ((newval.view(-1) - b_ret[mb]) ** 2).mean()
                loss    = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * vf_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_t
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/SPS",           sps,                global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss",   pg_loss.item(),     global_step)
        writer.add_scalar("losses/value_loss",    vf_loss.item(),     global_step)
        writer.add_scalar("losses/approx_kl",     approx_kl.item(),   global_step)
        writer.add_scalar("losses/clipfrac",      np.mean(clipfracs), global_step)
        writer.add_scalar("time/rollout_fps",     args.num_envs * args.num_steps / rollout_time, global_step)
        print(f"  SPS={sps}  pg={pg_loss.item():.4f}  vf={vf_loss.item():.4f}  kl={approx_kl.item():.4f}")

        if iteration % args.eval_freq == 1:
            _eval_run[0] += 1
            print("  [eval]")
            agent.eval()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, _, _, _, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v[mask].float())
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                writer.add_scalar(f"eval/{k}", mean, global_step)
                print(f"    eval_{k}={mean:.4f}")

        if args.save_model and iteration % args.eval_freq == 1:
            os.makedirs(run_dir, exist_ok=True)
            torch.save(agent.state_dict(), f"{run_dir}/ckpt_{iteration}.pt")

    if args.save_model:
        os.makedirs(run_dir, exist_ok=True)
        path = f"{run_dir}/final_ckpt.pt"
        torch.save(agent.state_dict(), path)
        print(f"Saved to {path}")

    writer.close()
    envs.close()
    eval_envs.close()