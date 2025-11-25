from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from envs.base_env import BaseEnv, Box


class UnityEnv(BaseEnv):
    def __init__(self, executable_path, decision_interval=1, render=True, max_episode_steps=1000):
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

    def _compute_reward(self, obs, next_obs, action, unity_reward, done):
        # 根據 CollectObservations 的順序解碼
        height = next_obs[0]           # hips 高度
        vel = next_obs[1:4]            # hips 局部速度 (x,y,z)
        up = next_obs[4:7]             # hips.up
        # fwd = next_obs[7:10]         # 站立不需要前進，就先不用

        # up 是一個向量，理想是 (0,1,0)，所以 up[1] 越接近 1 越直
        upright = up[1]

        # 盡量站著不亂晃：速度越大越扣分
        vel_penalty = -0.1 * float(np.linalg.norm(vel))

        # 動作過大也扣一點
        action_penalty = -0.001 * float(np.sum(np.square(action)))

        # 高度獎勵：越高越好（你之後可以 clamp 或縮放）
        height_reward = height

        # 總 reward：站高 + 站直 + 不亂晃 + 動作小
        reward = height_reward + 0.5 * upright + vel_penalty + action_penalty

        # 目前完全忽略 unity_reward
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

        reward = self._compute_reward(self._last_obs, next_obs, action, unity_reward, done)

        self._last_obs = next_obs

        return next_obs, reward, done, {}


    def close(self):
        self.env.close()
