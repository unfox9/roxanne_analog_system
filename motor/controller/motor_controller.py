import numpy as np
from motor.cpg.biped_cpg import BipedCPG
from motor.decoder.joint_mapper import JointMapper
from motor.decoder.joint_mapping_loader import load_action_channels

class MotorController:
    def __init__(self, mapping_path, dt=0.02, expected_dim=None):
        channels = load_action_channels(mapping_path)

        if expected_dim is not None and len(channels) != expected_dim:
            raise ValueError(f"mapping channels({len(channels)}) != expected_dim({expected_dim})")

        self.dt = dt
        self.cpg = BipedCPG(dt=dt)
        self.mapper = JointMapper(channels)
        self.reset()

    def reset(self):
        self.cpg.reset()

    def step(self, modulation):
        modulation = np.asarray(modulation, dtype=np.float32)
        cpg_state = self.cpg.step(modulation)
        joint_targets = self.mapper.decode(cpg_state)
        return joint_targets.astype(np.float32)
