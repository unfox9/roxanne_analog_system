from mlagents_envs.environment import UnityEnvironment
from base_env import BaseEnv

class UnityEnv(BaseEnv):
    def __init__(self, executable_path, decision_interval, render):
        self.env = UnityEnvironment(file_name=executable_path, no_graphics=not render)
        self.env.reset()

    def reset(self):
        obs = ...
        return obs

    def step(self, action):
        obs, reward, done = ...
        return obs, reward, done, {}

    def close(self):
        self.env.close()
