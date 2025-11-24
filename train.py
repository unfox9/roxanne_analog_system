from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import torch
import torch.nn as nn

# === init unity ===
env = UnityEnvironment(file_name=None, timeout_wait=60)
print("Created env, waiting for Unity...")
env.reset()
print("Env reset OK")

behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]
obs_dim = sum(np.prod(obs_spec.shape) for obs_spec in spec.observation_specs)
act_dim = spec.action_spec.continuous_size
print("obs_dim", obs_dim)
print("act_dim:", act_dim)
print("Behavior:", behavior_name)

# === Reward and obs ===
def my_reward(prev_obs, obs):
    
    last = obs[-10:] 

    height = last[0]              # hips.transform.position.y
    vel = last[1:4]               # local velocity (vx, vy, vz)
    up = last[4:7]                # hips.transform.up (ux, uy, uz)
    fwd = last[7:10]              # hips.transform.forward

    upright = up[1]              # world up = (0,1,0)

    reward = 0.0
    reward += upright * 0.01
    
    return reward


# === episodes ===
for episode in range(10):
    print(f"\n=== Episode {episode} ===")
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    step_idx = 0
    prev_obs = None
    episode_return = 0.0

    while len(terminal_steps) == 0:
        step_idx += 1

        # get obs
        obs_list = []
        for agent_id, step in decision_steps.items():
            obs = np.concatenate([o for o in step.obs], axis=0)
            obs_list.append(obs)
        obs_current = obs_list[0]

        # reward
        reward_py = my_reward(prev_obs, obs_current)
        episode_return += reward_py

        print(f"step {step_idx}, reward_py = {reward_py:.4f}")

        # action
        obs_batch = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        with torch.no_grad():
            action_batch = policy(obs_batch).cpu().numpy()

        action_tuple = ActionTuple(continuous=action_batch)
        env.set_actions(behavior_name, action_tuple)

        prev_obs = obs_current
    
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

    print(f"Episode {episode} return={episode_return:.3f}")

env.close()
