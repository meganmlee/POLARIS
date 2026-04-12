from __future__ import annotations
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import gymnasium as gym
import mani_skill.envs
import numpy as np
from ..adapters import BaseTaskAdapter

_POLARIS_MANIP_CACHE_ATTR = "_polaris_tcp_manip_cache"

# Must match skills/reach/reach_rrt.py:solve_ik (SAPIEN compact Pinocchio link index for panda_hand_tcp)
_DEFAULT_TCP_LINK_IDX = 10


@dataclass
class _TcpManipCache:
    """One SAPIEN PinocchioModel per robot — creating a new model every step is unsafe (crash / heap corruption)."""

    pmodel: Any
    tcp_link_idx: int = _DEFAULT_TCP_LINK_IDX


def _get_tcp_manip_cache(robot: Any) -> _TcpManipCache:
    c = getattr(robot, _POLARIS_MANIP_CACHE_ATTR, None)
    if isinstance(c, _TcpManipCache):
        return c
    scene = getattr(robot, "scene", None)
    if scene is not None and getattr(scene, "gpu_sim_enabled", False):
        raise RuntimeError(
            "TCP manipulability cache needs CPU simulation (create_pinocchio_model is disabled on GPU)."
        )
    pmodel = robot.create_pinocchio_model()
    cache = _TcpManipCache(pmodel=pmodel, tcp_link_idx=_DEFAULT_TCP_LINK_IDX)
    setattr(robot, _POLARIS_MANIP_CACHE_ATTR, cache)
    return cache


def _tcp_world_position(pmodel: Any, q_full: np.ndarray, tcp_link_idx: int) -> np.ndarray:
    """TCP linear position in world frame using only SAPIEN PinocchioModel (no separate pinocchio import)."""
    q_full = np.ascontiguousarray(np.asarray(q_full, dtype=np.float64).reshape(-1))
    pmodel.compute_forward_kinematics(q_full)
    pose = pmodel.get_link_pose(tcp_link_idx)
    p = getattr(pose, "p", pose)
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    return np.asarray(p, dtype=np.float64).reshape(3)


def _manip_from_linear_jacobian(J: np.ndarray) -> float:
    """Scalar manipulability: w = sqrt(det(J @ J.T)) for J in R^{3 x k} (position part)."""
    gram = J @ J.T
    det = float(np.linalg.det(gram))
    return float(np.sqrt(max(det, 0.0)))


def _numerical_position_jacobian(
    pmodel: Any,
    q_full: np.ndarray,
    tcp_link_idx: int,
    arm_dofs: int,
    eps: float,
) -> np.ndarray:
    """3 x arm_dofs ∂x_tcp / ∂q_arm via central-ish one-sided FD (stable, matches SAPIEN FK only)."""
    q_full = np.asarray(q_full, dtype=np.float64).reshape(-1).copy()
    p0 = _tcp_world_position(pmodel, q_full, tcp_link_idx)
    J = np.zeros((3, arm_dofs), dtype=np.float64)
    for i in range(arm_dofs):
        dq = q_full.copy()
        dq[i] += eps
        p1 = _tcp_world_position(pmodel, dq, tcp_link_idx)
        J[:, i] = (p1 - p0) / eps
    return J


def unwrap_maniskill_root(env: Any) -> Any:
    """Walk wrappers until we find a ManiSkill-style env with agent.robot."""
    seen: set[int] = set()
    e: Any = env
    for _ in range(32):
        if id(e) in seen:
            break
        seen.add(id(e))
        u = getattr(e, "unwrapped", e)
        agent = getattr(u, "agent", None)
        if agent is not None and getattr(agent, "robot", None) is not None:
            return u
        if hasattr(e, "env") and getattr(e, "env", None) is not e:
            e = e.env
        elif hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        else:
            break
    raise RuntimeError("unwrap_maniskill_root: could not find env.agent.robot")


def tcp_manipulability(
    robot: Any,
    tcp_frame_name: str = "panda_hand_tcp",
    *,
    arm_dofs: int = 7,
    fd_eps: float = 1e-4,
) -> np.ndarray:
    
    del tcp_frame_name  # API compatibility; index fixed to match IK in reach_rrt

    cache = _get_tcp_manip_cache(robot)
    pmodel, tcp_link_idx = cache.pmodel, cache.tcp_link_idx

    q_t = robot.qpos
    if hasattr(q_t, "detach"):
        q_t = q_t.detach()
    if hasattr(q_t, "cpu"):
        q_t = q_t.cpu().numpy()
    q_all = np.ascontiguousarray(np.asarray(q_t, dtype=np.float64))
    if q_all.ndim == 1:
        q_all = q_all.reshape(1, -1)

    ws: list[float] = []
    for i in range(q_all.shape[0]):
        q_row = q_all[i]
        J = _numerical_position_jacobian(pmodel, q_row, tcp_link_idx, arm_dofs, fd_eps)
        ws.append(_manip_from_linear_jacobian(J))
    return np.asarray(ws, dtype=np.float64)


def _load_reach_rrt():
    root = Path(__file__).resolve().parent.parent.parent
    path = root / "skills" / "reach" / "reach_rrt.py"
    spec = importlib.util.spec_from_file_location("polaris_reach_rrt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reach_rrt from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _q_full_from_arm(robot: Any, q7: np.ndarray) -> np.ndarray:
    q_t = robot.qpos
    if hasattr(q_t, "detach"):
        q_t = q_t.detach()
    if hasattr(q_t, "cpu"):
        q_t = q_t.cpu().numpy()
    q_full = np.asarray(q_t, dtype=np.float64).reshape(-1).copy()
    q7 = np.asarray(q7, dtype=np.float64).reshape(-1)
    q_full[: min(7, len(q_full))] = q7[: min(7, len(q_full))]
    return q_full


def _manip_from_q7(robot: Any, q7: np.ndarray) -> float:
    cache = _get_tcp_manip_cache(robot)
    q_full = _q_full_from_arm(robot, q7)
    J = _numerical_position_jacobian(
        cache.pmodel, q_full, cache.tcp_link_idx, min(7, q_full.shape[0]), 1e-4
    )
    return float(_manip_from_linear_jacobian(J))


def _tcp_from_q7(robot: Any, q7: np.ndarray) -> np.ndarray:
    cache = _get_tcp_manip_cache(robot)
    q_full = _q_full_from_arm(robot, q7)
    return _tcp_world_position(cache.pmodel, q_full, cache.tcp_link_idx)


def _tcp_obstacle_clearance(root: Any, tcp_pos: np.ndarray) -> float:
    tcp = np.asarray(tcp_pos, dtype=np.float64).reshape(3)
    obstacles = getattr(root, "obstacles", None)
    if not obstacles:
        return 1.0
    specs = getattr(root, "OBSTACLE_SPECS", None)
    if specs is None:
        specs = [(0.02, []), (0.015, []), (0.025, []), (0.018, [])]
    dmin = float("inf")
    for i, o in enumerate(obstacles):
        half = float(specs[i][0]) if i < len(specs) else 0.02
        p = o.pose.p
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        c = np.asarray(p, dtype=np.float64).reshape(3)
        dd = np.maximum(np.abs(tcp - c) - half, 0.0)
        dmin = min(dmin, float(np.linalg.norm(dd)))
    return float(dmin) if dmin < float("inf") else 1.0


def _normalize_triplet(
    manip_vals: list[float],
    clear_vals: list[float],
    dist_start: float,
    dist_end: float,
    *,
    manip_scale: float = 0.05,
    clear_scale: float = 0.25,
) -> Tuple[float, float, float]:
    if not manip_vals:
        m_n = 0.0
    else:
        m_n = float(np.clip(np.mean(manip_vals) / manip_scale, 0.0, 1.0))
    if not clear_vals:
        c_n = 0.0
    else:
        c_n = float(np.clip(np.mean(clear_vals) / clear_scale, 0.0, 1.0))
    ds = max(float(dist_start), 1e-9)
    p_n = float(np.clip((dist_start - dist_end) / ds, 0.0, 1.0))
    return m_n, p_n, c_n


def weighted_reach_score(
    manip_vals: list[float],
    clear_vals: list[float],
    dist_start: float,
    dist_end: float,
    w_manip: float,
    w_prog: float,
    w_clear: float,
) -> Tuple[float, float, float, float]:
    m_n, p_n, c_n = _normalize_triplet(manip_vals, clear_vals, dist_start, dist_end)
    s = w_manip * m_n + w_prog * p_n + w_clear * c_n
    return s, m_n, p_n, c_n


def lookahead_planner_score(
    root: Any,
    robot: Any,
    goal_pos: np.ndarray,
    q_start: np.ndarray,
    *,
    preview_steps: int = 24,
    w_manip: float = 1.0 / 3.0,
    w_prog: float = 1.0 / 3.0,
    w_clear: float = 1.0 / 3.0,
    rrt_max_iter: int = 600,
    rrt_step: float = 0.1,
    rrt_goal_thr: float = 0.05,
) -> Tuple[float, dict]:
    rr = _load_reach_rrt()
    goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]

    rs = np.random.get_state()
    try:
        np.random.seed(0)
        q_goal = rr.solve_ik(root, goal_pos, q_start)
        if q_goal is None:
            return float("-inf"), {"reason": "ik_failed"}
        planner = rr.RRTConnect(
            q_start,
            q_goal,
            max_iter=rrt_max_iter,
            step_size=rrt_step,
            goal_threshold=rrt_goal_thr,
        )
        path = planner.plan()
        n_pts = max(preview_steps, min(150, len(path) * 15))
        traj = np.asarray(rr.smooth_path_spline(path, num_points=n_pts), dtype=np.float32)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
    finally:
        np.random.set_state(rs)

    k = min(preview_steps, int(traj.shape[0]))
    if k < 1:
        return float("-inf"), {"reason": "empty_traj"}

    manip_vals: list[float] = []
    clear_vals: list[float] = []
    tcp0 = _tcp_from_q7(robot, traj[0])
    dist_start = float(np.linalg.norm(tcp0 - goal_pos))
    for i in range(k):
        q = np.asarray(traj[i], dtype=np.float32).reshape(-1)[:7]
        try:
            manip_vals.append(_manip_from_q7(robot, q))
        except Exception:
            manip_vals.append(0.0)
        tcp = _tcp_from_q7(robot, q)
        clear_vals.append(_tcp_obstacle_clearance(root, tcp))
    dist_end = float(np.linalg.norm(_tcp_from_q7(robot, traj[k - 1]) - goal_pos))

    s, m_n, p_n, c_n = weighted_reach_score(
        manip_vals, clear_vals, dist_start, dist_end, w_manip, w_prog, w_clear
    )
    return s, {"manip_norm": m_n, "progress_norm": p_n, "clearance_norm": c_n}


def _info_dist(info: Any) -> float:
    if not isinstance(info, dict):
        return float("nan")
    d = info.get("dist_to_goal")
    if d is None:
        return float("nan")
    return float(np.asarray(d).reshape(-1)[0])


def _dist_tcp_to_goal(obs: Dict[str, Any], goal_pos: np.ndarray) -> float:
    g = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    ee = obs.get("extra", {}) if isinstance(obs, dict) else {}
    if not isinstance(ee, dict):
        return float("nan")
    if "ee_pos" in ee:
        ep = np.asarray(ee["ee_pos"], dtype=np.float64).reshape(-1)[:3]
        return float(np.linalg.norm(ep - g))
    if "tcp_pose" in ee:
        ep = np.asarray(ee["tcp_pose"], dtype=np.float64).reshape(-1)[:3]
        return float(np.linalg.norm(ep - g))
    return float("nan")


def lookahead_rl_score(
    wrapper: "ManiSkillPlanningWrapper",
    goal_pos: np.ndarray,
    policy_act: Callable[[Dict[str, Any]], Any],
    obs: Dict[str, Any],
    *,
    preview_steps: int = 24,
    w_manip: float = 1.0 / 3.0,
    w_prog: float = 1.0 / 3.0,
    w_clear: float = 1.0 / 3.0,
    dist_start_override: Optional[float] = None,
) -> Tuple[float, dict]:
    root = wrapper.root
    robot = wrapper.robot
    if robot is None:
        return float("-inf"), {"reason": "no_robot"}

    goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    if dist_start_override is not None:
        dist_start = float(dist_start_override)
    else:
        dist_start = _dist_tcp_to_goal(obs, goal_pos)
    snapshot = wrapper.clone_state()
    manip_vals: list[float] = []
    clear_vals: list[float] = []

    try:
        o = obs
        info: Dict[str, Any] = {}
        for _ in range(preview_steps):
            act = policy_act(o)
            if hasattr(act, "detach"):
                act = act.detach().cpu().numpy()
            act = np.asarray(act, dtype=np.float32).reshape(-1)
            o, _, term, trunc, info = wrapper.env.step(act)
            if isinstance(info, dict) and "dist_to_goal" not in info and hasattr(root, "evaluate"):
                try:
                    ev = root.evaluate()
                    dg = ev.get("dist_to_goal")
                    if dg is not None:
                        info = {**info, "dist_to_goal": dg}
                except Exception:
                    pass
            try:
                w = tcp_manipulability(robot)
                manip_vals.append(float(np.mean(w)))
            except Exception:
                manip_vals.append(0.0)
            ee = o.get("extra", {}) if isinstance(o, dict) else {}
            if isinstance(ee, dict) and "ee_pos" in ee:
                tcp = np.asarray(ee["ee_pos"], dtype=np.float64).reshape(-1)[:3]
            elif isinstance(ee, dict) and "tcp_pose" in ee:
                tcp = np.asarray(ee["tcp_pose"], dtype=np.float64).reshape(-1)[:3]
            else:
                q_t = robot.qpos
                if hasattr(q_t, "detach"):
                    q_t = q_t.detach().cpu().numpy()
                tcp = _tcp_from_q7(robot, np.asarray(q_t, dtype=np.float64).reshape(-1)[:7])
            clear_vals.append(_tcp_obstacle_clearance(root, tcp))
            if np.asarray(term).any() or np.asarray(trunc).any():
                break
        dist_end = _info_dist(info)
        if dist_end != dist_end:
            dist_end = _dist_tcp_to_goal(o, goal_pos)
        if dist_start != dist_start:
            dist_start = dist_end if dist_end == dist_end else 1.0
        if dist_end != dist_end:
            dist_end = dist_start if dist_start == dist_start else 1.0
        if dist_start != dist_start:
            dist_start = 1.0
        if dist_end != dist_end:
            dist_end = 1.0
    finally:
        wrapper.restore_state(snapshot)

    s, m_n, p_n, c_n = weighted_reach_score(
        manip_vals, clear_vals, dist_start, dist_end, w_manip, w_prog, w_clear
    )
    return s, {"manip_norm": m_n, "progress_norm": p_n, "clearance_norm": c_n}


def select_reach_backend(
    planner_score: float,
    rl_score: float,
) -> str:
    if planner_score != planner_score and rl_score == rl_score:
        return "rl"
    if rl_score != rl_score and planner_score == planner_score:
        return "planner"
    return "planner" if float(planner_score) >= float(rl_score) else "rl"



class ManiSkillPlanningWrapper:
    def __init__(self, env: Any, adapter: Optional[BaseTaskAdapter] = None, hide_obj_orientation: bool = False):
        self.env = env
        self.adapter = adapter
        self.hide_obj_orientation = hide_obj_orientation
        self.root = self.env.unwrapped
        self.agent = getattr(self.root, "agent", None)

        self.controller = getattr(self.agent, "controller", None) if self.agent else None
        self.robot = getattr(self.agent, "robot", None) if self.agent else None
    
    def clone_state(self) -> Dict[str, Any]:
        state = self.root.get_state_dict()

        sim_state: Dict[str, Any] = {}
        if "actors" in state:
            sim_state["actors"] = state["actors"]
        if "articulations" in state:
            sim_state["articulations"] = state["articulations"]
        
        controller_state = state.get("controller", None)
        
        if self.adapter is not None:
            task_state = self.adapter.get_task_state(self.root)
        else:
            task_state = None
        snapshot = {
            "sim_state": sim_state,
            "controller_state": controller_state,
            "task_state": task_state,
        }
        return snapshot

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        
        sim_state = snapshot.get("sim_state", None)
        controller_state = snapshot.get("controller_state", None)
        task_state = snapshot.get("task_state", None)

        if sim_state is not None:
            self.root.set_state_dict(sim_state)
        if controller_state is not None and self.agent is not None:
            if hasattr(self.agent, "set_controller_state"):
                self.agent.set_controller_state(controller_state)
            else:
                pass
        
        if task_state is not None and self.adapter is not None:
            self.adapter.set_task_state(self.root, task_state)
        else:
            pass

    def get_planning_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if not isinstance(obs, dict):
            raise TypeError(f"get_planning_obs expects dict obs, got {type(obs)}")
        
        agent = obs.get("agent")
        extra = obs.get("extra")

        if agent is None or extra is None:
            raise KeyError("Expected obs to contain 'agent' and 'extra' keys.")
        if "qpos" not in agent or "qvel" not in agent:
            raise KeyError("obs['agent'] must contain 'qpos' and 'qvel' for planning_obs.")

        qpos = np.asarray(agent["qpos"], dtype=np.float32).copy()
        qvel = np.asarray(agent["qvel"], dtype=np.float32).copy()

        # Required keys for planning observation
        required_keys = ["tcp_pose", "goal_pos", "obj_pose"]
        missing_extra = [k for k in required_keys if k not in extra]
        if missing_extra:
            env_name = getattr(self.root, "spec", None)
            env_id = getattr(env_name, "id", None) if env_name else None
            env_hint = f" for {env_id}" if env_id else ""
            raise KeyError(
                f"obs['extra'] missing required keys for planning_obs: {missing_extra}. "
                f"Make sure obs_mode='state_dict'{env_hint}."
            )
        
        tcp_pose = np.asarray(extra["tcp_pose"], dtype=np.float32).copy()
        goal_pos = np.asarray(extra["goal_pos"], dtype=np.float32).copy()
        obj_pose = np.asarray(extra["obj_pose"], dtype=np.float32).copy()
        
        planning_obs = {
            "qpos": qpos,
            "qvel": qvel,
            "tcp_pose": tcp_pose,
            "goal_pos": goal_pos,
            "obj_pose": obj_pose,
        }
        
        # Optionally include multi-object fields if they exist (for shelf retrieval)
        if "target_obj_pose" in extra:
            planning_obs["target_obj_pose"] = np.asarray(extra["target_obj_pose"], dtype=np.float32).copy()
        if "obj_poses" in extra:
            planning_obs["obj_poses"] = np.asarray(extra["obj_poses"], dtype=np.float32).copy()
        if "target_obj_id" in extra:
            planning_obs["target_obj_id"] = extra["target_obj_id"]

        return planning_obs
    
    def flatten_planning_obs(self, planning_obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(planning_obs["qpos"], dtype=np.float32).ravel(),
                np.asarray(planning_obs["qvel"], dtype=np.float32).ravel(),
                np.asarray(planning_obs["obj_pose"], dtype=np.float32).ravel(),
                np.asarray(planning_obs["goal_pos"], dtype=np.float32).ravel(),
                np.asarray(planning_obs["tcp_pose"], dtype=np.float32).ravel(),
            ],
            axis=0,
        )
    
    def get_qpos(self) -> np.ndarray:
        if self.robot is None or not hasattr(self.robot, "get_qpos"):
            raise RuntimeError("Robot articulation missing or does not implement get_qpos().")
        qpos = self.robot.get_qpos()
        return np.asarray(qpos, dtype=np.float32).copy()
    
    def get_qvel(self) -> np.ndarray:
        if self.robot is None or not hasattr(self.robot, "get_qvel"):
            raise RuntimeError("Robot articulation missing or does not implement get_qvel().")
        qvel = self.robot.get_qvel()
        return np.asarray(qvel, dtype=np.float32).copy()
    
    def controlled_joint_indices(self) -> np.ndarray:
        if self.controller is None:
            raise RuntimeError("Controller not found on agent.")
        
        idx = getattr(self.controller, "active_joint_indices", None)
        if idx is None:
            raise RuntimeError(
                "Controller does not expose 'active_joint_indices'. "
                "Check the ManiSkill controller API."
            )
        return np.asarray(idx, dtype=np.int64).copy()
    
    def controlled_qpos(self) -> np.ndarray:
        if self.controller is None:
            raise RuntimeError("Controller not found on agent.")
        
        if hasattr(self.controller, "qpos"):
            qpos = np.array(self.controller.qpos, dtype=np.float32)
            return qpos.copy()
        
        full_qpos = self.get_qpos()
        idx = self.controlled_joint_indices()

        return full_qpos[idx]
    
    def controlled_qvel(self) -> np.ndarray:
        if self.controller is None:
            raise RuntimeError("Controller not found on agent.")
        
        if hasattr(self.controller, "qvel"):
            qvel = np.array(self.controller.qvel, dtype=np.float32)
            return qvel.copy()
        
        full_qvel = self.get_qvel()
        idx = self.controlled_joint_indices()
        
        return full_qvel[idx]
    def print_controller_summary(self) -> None:
        print("\n=== Controller / Robot Summary ===")

        if self.robot is None:
            print("Robot: <missing>")
        else:
            # DOF and joint names from articulation
            qpos = self.get_qpos()
            dof = qpos.shape[0]
            print(f"Robot DOF: {dof}")

            joint_names = []
            if hasattr(self.robot, "get_joints"):
                try:
                    joints = self.robot.get_joints()
                    for j in joints:
                        # SAPIEN joints usually have 'get_name()'
                        name = getattr(j, "get_name", None)
                        if callable(name):
                            joint_names.append(name())
                        elif hasattr(j, "name"):
                            joint_names.append(j.name)
                except Exception:
                    pass

            if joint_names:
                print("Joints:")
                for i, n in enumerate(joint_names):
                    print(f"  [{i}] {n}")
            else:
                print("Joints: <could not retrieve joint names>")

        if self.controller is None:
            print("\nController: <missing>")
            return

        # Controlled joints
        try:
            idx = self.controlled_joint_indices()
            print("\nControlled joint indices:", idx.tolist())
        except Exception as e:
            print("\nControlled joint indices: <error>", e)

        # Controller repr usually contains joint + DOF info
        print("\nController repr:")
        print(self.controller)
        print("=== End summary ===\n")

    def _filter_obs(self, obs):
        if not self.hide_obj_orientation:
            return obs
        if not isinstance(obs, dict) or "extra" not in obs:
            return obs
        extra = obs.get("extra", None)
        if not isinstance(extra, dict) or "obj_pose" not in extra:
            return obs

        obj_pose = np.asarray(extra["obj_pose"], dtype=np.float32).copy()
        if obj_pose.shape[-1] >= 7:
            obj_pose[..., 3:7] = np.array([0, 0, 0, 1], dtype=np.float32)

        # write back (copy dicts so you don't mutate shared references)
        obs2 = dict(obs)
        extra2 = dict(extra)
        extra2["obj_pose"] = obj_pose
        obs2["extra"] = extra2
        return obs2
        
    def reset(self, *args, **kwargs):
        # Use the wrapped env to respect outer wrappers (viewer/step hooks, etc.)
        obs, info = self.env.reset(*args, **kwargs)
        return self._filter_obs(obs), info
    
    def step(self, action: np.ndarray):
        # Step through the wrapped env (not the unwrapped root) for wrapper behavior
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._filter_obs(obs), reward, terminated, truncated, info
    
    def close(self):
        # Close via the wrapped env to respect any outer wrapper hooks
        return self.env.close()

    def get_controller_bounds(self) -> tuple:
        """
        Get the original (unnormalized) action bounds for the controller.
        Returns (low, high) tuple for position control bounds.
        For pd_ee_delta_pose, this is typically (-0.1, 0.1) for position.
        """
        if self.controller is None:
            raise RuntimeError("Controller not found on agent.")
        
        # Try to get bounds from the arm controller if it's a CombinedController
        if hasattr(self.controller, "controllers") and "arm" in self.controller.controllers:
            arm_controller = self.controller.controllers["arm"]
            if hasattr(arm_controller, "action_space_low") and hasattr(arm_controller, "action_space_high"):
                # Get position bounds (first 3 dims)
                low = arm_controller.action_space_low[0:3].cpu().numpy()
                high = arm_controller.action_space_high[0:3].cpu().numpy()
                # For pd_ee_delta_pose, bounds should be the same for all 3 dims
                return (float(low[0]), float(high[0]))
        
        # Default bounds for pd_ee_delta_pose (from panda_stick config)
        return (-0.1, 0.1)
    
    def __getattr__(self, name):
        return getattr(self.root, name)
    
def main():
    env = gym.make(
        "PushT-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        )
    wrapper = ManiSkillPlanningWrapper(env)
    snapshot = wrapper.clone_state()
    wrapper.restore_state(snapshot)
    # from planning_wrapper.adapters.pusht import PushTTaskAdapter

    # env = gym.make("PushT-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose")
    # adapter = PushTTaskAdapter()
    # w = MSPlanningWrapper(env, adapter=adapter)

    # snap = w.clone_state()
    # # snap now contains "task_state"
    # print(snap["task_state"].keys())  # e.g. goal_tee_pose, ee_goal_pose

    # w.restore_state(snap)
    # print(wrapper.root.observation_space)
    # print(type(wrapper.env))
    # print(type(wrapper.root))
    # print(type(wrapper.agent))
    # print(type(wrapper.controller))
    # print(type(wrapper.robot))
    # print("================================================")
    # print(wrapper.root)
    # print(wrapper.agent)
    # print(wrapper.controller)
    # print(wrapper.robot)
    # print("================================================")   
    # print("root:", type(wrapper.root))
    # print("agent:", type(wrapper.agent))
    # print("controller:", type(wrapper.controller))
    # print("robot:", type(wrapper.robot))
    # print("================================================")
    # obs, info = wrapper.reset(seed=0)
    # print("reset ok")
    # print("================================================")
    # action = wrapper.action_space.sample()
    # print("action:", action)
    # print("================================================")
    # obs, reward, terminated, truncated, info = wrapper.step(action)
    # print("step ok")
    # print("obs:", obs)
    # print("reward:", reward)
    # print("terminated:", terminated)
    # print("truncated:", truncated)
    # print("info:", info)
    # print("================================================")
    # wrapper.close()
    # print("close ok")
    # print("================================================")
if __name__ == "__main__":
    main()
    