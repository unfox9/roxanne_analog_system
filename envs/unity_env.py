from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from envs.base_env import BaseEnv, Box


class UnityEnv(BaseEnv):
    def __init__(self, executable_path, decision_interval=5, render=True, max_episode_steps=5000):
        self.env = UnityEnvironment(file_name=None, no_graphics=not render)
        self.env.reset()

        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        self.decision_interval = decision_interval
        self._max_episode_steps = max_episode_steps

        # 觀測空間
        obs_dim = sum(int(np.prod(obs_spec.shape)) for obs_spec in self.spec.observation_specs)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # 動作空間（假設只有連續動作）
        action_dim = self.spec.action_spec.continuous_size
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

    def _get_obs_from_steps(self, decision_steps):
        obs_list = []
        for arr in decision_steps.obs:
            # arr: (n_agents, ...) -> 取第 0 個 agent，然後展平成向量
            obs_list.append(arr[0].ravel())
        obs = np.concatenate(obs_list, axis=0)
        return obs.astype(np.float32)

    def reset(self):
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        obs = self._get_obs_from_steps(decision_steps)
        return obs

    def _compute_reward(self, next_obs, action, unity_reward, done):
        height = next_obs[0]
        vel = next_obs[1:4]
        up = next_obs[4:7]

        # scale up
        height_term = 5.0 * (height - 0.5)   # 最高大約 +3
        upright_term = 5.0 * up[1]           # 0~5
        vel_penalty = -1.0 * np.linalg.norm(vel)

        reward = height_term + upright_term + vel_penalty

        # 加一點 Unity 的 reward
        reward += unity_reward * 0.5

        return reward

    def step(self, action):
        action = np.array(action, dtype=np.float32).reshape(1, -1)
        action_tuple = ActionTuple(continuous=action)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            step = terminal_steps
            done = True
        else:
            step = decision_steps
            done = False

        next_obs = self._get_obs_from_steps(step)
        unity_reward = float(step.reward[0])

        reward = self._compute_reward(next_obs, action, unity_reward, done)

        return next_obs, reward, done, {}


    def close(self):
        self.env.close()
