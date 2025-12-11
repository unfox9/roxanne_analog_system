import numpy as np
from envs.base_env import BaseEnv, Box

class MotorEnv(BaseEnv):
    def __init__(
        self,
        inner_env,
        controller_factory,
        latent_dim=8,
        dt=0.02,
    ):
        self.inner_env = inner_env  
        self.controller_factory = controller_factory
        self.dt = dt
        self.controllers = [
            controller_factory()
            for _ in range(self.inner_env.num_agents)
        ]
        self.observation_space = self.inner_env.observation_space
        self.latent_dim = latent_dim
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(latent_dim,),
            dtype=np.float32,
        )
        self.num_agents = self.inner_env.num_agents
        self._max_episode_steps = self.inner_env._max_episode_steps

    def reset(self):
        obs = self.inner_env.reset()
        self.controllers = [
            self.controller_factory()
            for _ in range(self.inner_env.num_agents)
        ]
        return obs

    def step(self, latent_action):
        latent_action = np.asarray(latent_action, dtype=np.float32)

        if latent_action.shape == (self.latent_dim,):
            latent_action = latent_action[np.newaxis, :]

        assert latent_action.shape[0] == self.inner_env.num_agents, \
            f"latent_action num_agents {latent_action.shape[0]} " \
            f"!= inner_env.num_agents {self.inner_env.num_agents}"

        full_actions = []
        for i in range(self.inner_env.num_agents):
            a_full = self.controllers[i].step(latent_action[i])
            full_actions.append(a_full)

        full_actions = np.stack(full_actions, axis=0).astype(np.float32)

        next_obs, reward, done, info = self.inner_env.step(full_actions)
        return next_obs, reward, done, info

    def close(self):
        self.inner_env.close()
