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

        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self.agent_ids = list(decision_steps.agent_id)
        self.num_agents = len(decision_steps)
        self.id_to_index = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}

        self.last_decision_agent_ids = list(decision_steps.agent_id)

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

    def _build_obs_reward_done(self, decision_steps, terminal_steps):
        obs = np.zeros((self.num_agents, self.total_obs_dim), dtype=np.float32)
        reward = np.zeros((self.num_agents,), dtype=np.float32)
        done = np.zeros((self.num_agents,), dtype=bool)


        def fill_from_steps(steps, is_termial: bool):
            for step_idx, agent_id in enumerate(steps.agent_id):
                agent_id_int = int(agent_id)

                if agent_id_int not in self.id_to_index:
                    new_idx = len(self.id_to_index)
                    if new_idx >= self.num_agents:
                        raise ValueError("New agent appeared beyond initial num_agents.")
                    
                    self.id_to_index[agent_id_int] = new_idx
                
                idx = self.id_to_index[agent_id_int]

                per_agent_obs = []
                for arr in steps.obs:
                    per_agent_obs.append(arr[step_idx].reshape(-1))
                obs[idx] = np.concatenate(per_agent_obs)

                reward[idx] = steps.reward[step_idx]
                if is_termial:
                    done[idx] = 1.0

        fill_from_steps(decision_steps, is_termial=False)
        fill_from_steps(terminal_steps, is_termial=True)
                
        return obs, reward, done

    def reset(self):
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        self.agent_ids = list(decision_steps.agent_id)
        self.num_agents = len(decision_steps)
        self.id_to_index = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        self.last_decision_agent_ids = list(decision_steps.agent_id)

        obs = np.zeros((self.num_agents, self.total_obs_dim), dtype=np.float32)
        for i, agent_id in enumerate(decision_steps.agent_id):
            idx = self.id_to_index[agent_id]
            per_agent_obs = [arr[i].reshape(-1) for arr in decision_steps.obs]
            obs[idx] = np.concatenate(per_agent_obs)

        return obs


    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        if action.shape == (self.action_dim,):
            action = action[np.newaxis, :]
        assert action.shape == (self.num_agents, self.action_dim), \
            f"Action shape {action.shape} != ({self.num_agents}, {self.action_dim})"

        n_req = len(self.last_decision_agent_ids)

        action_to_send = np.zeros((len(self.last_decision_agent_ids), self.action_dim), dtype=np.float32)
        for i, agent_id in enumerate(self.last_decision_agent_ids):
            idx = self.id_to_index[agent_id]
            action_to_send[i] = action[idx]

        action_tuple = ActionTuple(continuous=action_to_send)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        next_obs, reward, done = self._build_obs_reward_done(decision_steps, terminal_steps)

        self.last_decision_agent_ids = list(decision_steps.agent_id)

        return next_obs, reward, done, {}


    def close(self):
        self.env.close()
