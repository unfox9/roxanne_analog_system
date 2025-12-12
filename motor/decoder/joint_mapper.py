import numpy as np
from motor.schema.action_schema import ActionSchema


class JointMapper:
    def __init__(self, schema: ActionSchema):
        self.schema = schema
        self.channels = schema.channels
        self.action_dim = schema.action_dim

        # 直接用 schema.groups（集中管理）
        self.left_leg = schema.groups["left_leg"]
        self.right_leg = schema.groups["right_leg"]
        self.spine = schema.groups["spine"]
        self.tail = schema.groups["tail"]
        self.extra = schema.groups["extra"]

    def decode(self, cpg_state) -> np.ndarray:
        out = np.zeros(self.action_dim, dtype=np.float32)

        left_wave = cpg_state["left"]
        right_wave = cpg_state["right"]
        spine_phase = cpg_state["left_phase"]
        spine_amp = cpg_state["spine_amp"]
        tail_amp = cpg_state["tail_amp"]

        for c in self.left_leg:
            out[c["index"]] = left_wave
        for c in self.right_leg:
            out[c["index"]] = right_wave

        for i, c in enumerate(self.spine):
            phase = spine_phase + i * 0.35
            out[c["index"]] = spine_amp * np.sin(phase)

        for i, c in enumerate(self.tail):
            phase = spine_phase + np.pi + i * 0.45
            out[c["index"]] = tail_amp * np.sin(phase)

        return out
