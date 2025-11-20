from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

env = UnityEnvironment(file_name=None, timeout_wait=60)
print("Created env, waiting for Unity...")
env.reset()
print("Env reset OK")

behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]

print("Behavior:", behavior_name)
print("Observation specs:")
for i, obs_spec in enumerate(spec.observation_specs):
    print(f"  obs[{i}] shape = {obs_spec.shape}")
print("Action spec:", spec.action_spec)

def my_reward_and_done(prev_obs, obs):
    # e.g. obs[0] = hips height (依你 CollectObservations 的順序)
    height = obs[0]              # hips.transform.position.y
    vel = obs[1:4]               # local velocity (vx, vy, vz)
    up = obs[4:7]                # hips.transform.up (ux, uy, uz)
    fwd = obs[7:10]              # hips.transform.forward

# 2. 直立程度：dot(hips.up, world_up) = up_y
    upright = up[1]              # world up = (0,1,0)

    reward = 0.0
    reward += upright * 0.01
    if height < 0.5:
        reward -= 1.0
        done = True
    else:
        done = False

    return reward, done

for episode in range(100):
    print(f"\n=== Episode {episode} ===")
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    step_idx = 0
    prev_obs = None
    episode_return = 0.0

    # 這裡我改成 while True，因為 done_py 可能跟 Unity 的 terminal 無關
    while True:
        step_idx += 1
        n_agents = len(decision_steps)
        act_dim = spec.action_spec.continuous_size

        # 先暫時用隨機動作
        actions = np.random.uniform(-1, 1, size=(n_agents, act_dim)).astype(np.float32)
        action_tuple = ActionTuple(continuous=actions)
        env.set_actions(behavior_name, action_tuple)

        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # 這裡假設只有一個 agent，拿第一個就好
        if len(decision_steps) > 0:
            agent_id = list(decision_steps.agent_id_to_index.keys())[0]
            step = decision_steps[agent_id]
        else:
            # 偶爾 Unity 可能會把 agent 放在 terminal_steps（如果你有 EndEpisode）
            agent_id = list(terminal_steps.agent_id_to_index.keys())[0]
            step = terminal_steps[agent_id]

        obs = step.obs[0]  # numpy array, shape = (obs_dim,)

        reward_py, done_py = my_reward_and_done(prev_obs, obs)
        episode_return += reward_py

        print(f"step {step_idx}, reward_py = {reward_py:.4f}, cum_return = {episode_return:.4f}")

        # 這裡你真正用來做 RL 的是 reward_py / done_py
        # step.reward（Unity reward）可以完全無視

        if done_py:
            print(f"episode {episode} finished at step {step_idx}, return = {episode_return:.4f}")
            break

        prev_obs = obs

env.close()
