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

for episode in range(10):
    print(f"\n=== Episode {episode} ===")
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    step_idx = 0
    while len(terminal_steps) == 0:
        step_idx += 1
        n_agents = len(decision_steps)
        act_dim = spec.action_spec.continuous_size

        # 隨機動作，先確認會不會動
        actions = np.random.uniform(-1, 1, size=(n_agents, act_dim)).astype(np.float32)
        action_tuple = ActionTuple(continuous=actions)
        env.set_actions(behavior_name, action_tuple)

        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # 簡單印一下 reward 看有沒有在跑
        for agent_id, step in decision_steps.items():
            print(f"step {step_idx}, reward = {step.reward}")

env.close()
