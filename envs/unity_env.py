from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from envs.base_env import BaseEnv, Box


class UnityEnv(BaseEnv):
    def __init__(self, executable_path, decision_interval=5, render=True, max_episode_steps=5000,
                 num_toes=4, min_upright=0.85, com_k=5.0):
        file_name = executable_path
        self.env = UnityEnvironment(file_name=file_name, no_graphics=not render)
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

        # 獎勵參數
        self.num_toes = num_toes
        self.min_upright = min_upright
        self.com_k = com_k



    def _get_obs_from_steps(self, decision_steps):
        obs_list = []
        for arr in decision_steps.obs:
            obs_list.append(arr[0].ravel())
        obs = np.concatenate(obs_list, axis=0)
        return obs.astype(np.float32)

    def reset(self):
        self.env.reset()
        self._episode_step = 0
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        obs = self._get_obs_from_steps(decision_steps)
        return obs

    def _compute_reward(self, next_obs, action, done):
        height = next_obs[0]
        vel = next_obs[1:4]
        up = next_obs[4:7]
        fwd = next_obs[7:10]
        com_local = next_obs[10:13] # center of mass in local frame

        up_y = up[1]

        com_xy = np.linalg.norm(next_obs[10:12])

        n_tose = self.num_toes
        tose = next_obs[-n_tose:]
        grounded_frac = np.mean(tose)

        r_upright = np.clip((up_y - self.min_upright) / (1.0 - self.min_upright), 0.0, 1.0)
        r_com = np.exp(-self.com_k * com_xy * com_xy)
        r_ground = grounded_frac

        reward = (
            0.4 * r_upright +
            0.4 * r_com +
            0.2 * r_ground
        )

        if done:
            reward -= 1.0

        return reward

    def step(self, action):
        action = np.array(action, dtype=np.float32).reshape(1, -1)
        action_tuple = ActionTuple(continuous=action)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            step = terminal_steps
            done_by_unity = True
        else:
            step = decision_steps
            done_by_unity = False

        self._episode_step += 1
        done_by_timeout = self._episode_step >= self._max_episode_steps
        
        done = done_by_unity or done_by_timeout

        next_obs = self._get_obs_from_steps(step)
        reward = self._compute_reward(next_obs, action, done)

        if done_by_timeout and not done_by_unity:
            self.env.reset()
            self._episode_step = 0

        return next_obs, reward, done, {}


    def close(self):
        self.env.close()
