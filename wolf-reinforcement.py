import torch
import torch.nn as nn
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),  # output in [-1,1]
        )

    def forward(self, x):
        return self.net(x)

env = UnityEnvironment(file_name="WolfEnv/WolfEnv.exe")
env.reset()
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]

obs_dim = sum(np.prod(shape) for shape in spec.observation_shapes)
act_dim = spec.action_spec.continuous_size

policy = PolicyNet(obs_dim, act_dim)

for episode in range(10):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    while len(terminal_steps) == 0:
        # 把 obs 拼成一個 tensor
        obs_list = []
        for agent_id, step in decision_steps.items():
            obs = np.concatenate([o for o in step.obs], axis=0)
            obs_list.append(obs)

        obs_batch = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        with torch.no_grad():
            actions_batch = policy(obs_batch).cpu().numpy()

        action_tuple = ActionTuple(continuous=actions_batch)
        env.set_actions(behavior_name, action_tuple)
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

env.close()
