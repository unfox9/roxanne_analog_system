from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from envs.base_env import BaseEnv, Box


class UnityEnv(BaseEnv):
    def __init__(self, executable_path, decision_interval=5, render=True, max_episode_steps=5000,):
        file_name = executable_path
        self.env = UnityEnvironment(file_name=file_name, no_graphics=not render)
        self.env.reset()

        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        self.decision_interval = decision_interval
        self._max_episode_steps = max_episode_steps

        # 一開始的 agents 數量
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self.agent_ids = list(decision_steps.agent_id)
        self.num_agents = len(decision_steps)
        self.id_to_index = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}

        self.last_decision_agent_ids = list(decision_steps.agent_id)

        self._episode_step = 0

        # 觀測空間
        obs_dim = sum(int(np.prod(obs_spec.shape)) for obs_spec in self.spec.observation_specs)
        self.total_obs_dim = obs_dim
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # 動作空間（假設只有連續動作）
        action_dim = self.spec.action_spec.continuous_size
        self.action_dim = action_dim
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

    def _get_obs_from_steps(self, steps):
        """
        把 DecisionSteps 或 TerminalSteps 轉成 (num_agents, obs_dim)，
        根據 self.id_to_index 把每個 agent 塞回自己的 row。
        沒出現在 steps 裡的 agent 會保持上一幀的值或 0（看你怎麼用）。
        """
        obs = np.zeros((self.num_agents, self.total_obs_dim), dtype=np.float32)
        for agent_id, index in self.id_to_index.items():
            if agent_id in steps.agent_id:
                step_index = list(steps.agent_id).index(agent_id)
                per_agent_obs = []
                for arr in steps.obs:
                    per_agent_obs.append(arr[step_index].reshape(-1))
                obs[index] = np.concatenate(per_agent_obs)
        return obs

    def _get_reward_from_steps(self, decision_steps, terminal_steps):
        """
        單步 reward：DecisionSteps 和 TerminalSteps 的 reward 都寫進同一個 array。
        如果同一個 agent 在 terminal_steps 裡，會覆蓋 decision_steps 裡的值。
        """
        reward = np.zeros(self.num_agents, dtype=np.float32)
        for steps in (decision_steps, terminal_steps):
            for i, agent_id in enumerate(steps.agent_id):
                reward[self.id_to_index[agent_id]] = steps.reward[i]
        return reward

    def reset(self):
        """
        重置整個 Unity 環境，重新抓 agent_ids / 對應關係。
        """
        self.env.reset()
        self._episode_step = 0

        decision_steps, _ = self.env.get_steps(self.behavior_name)

        self.agent_ids = list(decision_steps.agent_id)
        self.num_agents = len(decision_steps)
        self.id_to_index = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}
        self.last_decision_agent_ids = list(decision_steps.agent_id)

        # 重新設定 observation_space / action_space（多 agent 時數量可能變）
        obs_dim = sum(int(np.prod(obs_spec.shape)) for obs_spec in self.spec.observation_specs)
        self.total_obs_dim = obs_dim
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        action_dim = self.spec.action_spec.continuous_size
        self.action_dim = action_dim
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        obs = self._get_obs_from_steps(decision_steps)
        return obs

    def step(self, action):
        """
        多 agent 版本：
        - 外面給進來的 action shape = (num_agents, action_dim)
          或在 num_agents == 1 時給 (action_dim,)
        - 會自動依照上一個 decision_steps 的 agent 順序排好再丟給 Unity
        """
        action = np.asarray(action, dtype=np.float32)

        # 方便單 agent 使用：允許 (action_dim,) -> (1, action_dim)
        if action.shape == (self.action_dim,):
            action = action[np.newaxis, :]
        assert action.shape == (self.num_agents, self.action_dim), \
            f"Action shape {action.shape} != ({self.num_agents}, {self.action_dim})"

        # 把「固定 index 順序的 action」轉成「上一幀 decision_steps 的 agent 順序」
        action_to_send = np.zeros((len(self.last_decision_agent_ids), self.action_dim), dtype=np.float32)
        for i, agent_id in enumerate(self.last_decision_agent_ids):
            index = self.id_to_index[agent_id]
            action_to_send[i] = action[index]

        action_tuple = ActionTuple(continuous=action_to_send)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        done_by_unity = len(terminal_steps) > 0
        if len(terminal_steps) > 0:
            step_for_obs = terminal_steps
        else:
            step_for_obs = decision_steps

        self._episode_step += 1
        done_by_timeout = self._episode_step >= self._max_episode_steps

        done = bool(done_by_unity or done_by_timeout)

        next_obs = self._get_obs_from_steps(step_for_obs)
        reward = self._get_reward_from_steps(decision_steps, terminal_steps)
        #print("reward from unity:", reward)

        if done_by_timeout and not done_by_unity:
            next_obs = self.reset()

        if len(decision_steps) > 0:
            self.last_decision_agent_ids = list(decision_steps.agent_id)

        return next_obs, reward, done, {}


    def close(self):
        self.env.close()
