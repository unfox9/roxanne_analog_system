import mujoco
import mujoco.viewer
import numpy as np
import time

class WolfEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self.get_observation()

    def step(self, action):
        if action is not None:
            self.data.ctrl[:] = action

        mujoco.mj_step(self.model, self.data)
        
        return self.get_observation()

    def get_observation(self):
        return np.concatenate((self.data.qpos, self.data.qvel))

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        self.viewer.sync()
        
    def close(self):
        if self.viewer:
            self.viewer.close()