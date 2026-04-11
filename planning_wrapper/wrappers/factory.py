import gymnasium as gym
from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters.pusho import PushOTaskAdapter

def make_pusho_vis_gt():
    env = gym.make("PushT-v1", obs_mode="state_dict", control_mode="pd_joint_pos")
    return ManiSkillPlanningWrapper(env, adapter=PushOTaskAdapter(), hide_obj_orientation=False)

def make_pusho_vis_no_ori():
    env = gym.make("PushT-v1", obs_mode="state_dict", control_mode="pd_joint_pos")
    return ManiSkillPlanningWrapper(env, adapter=PushOTaskAdapter(), hide_obj_orientation=True)
