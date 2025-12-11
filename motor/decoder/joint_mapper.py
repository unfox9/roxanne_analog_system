import numpy as np

class JointMapper:
    def __init__(self, channels):
        self.channels = channels
        self.action_dim = len(channels)

        self.left_leg = [c for c in channels if "Left leg" in c["name"]]
        self.right_leg = [c for c in channels if "Right leg" in c["name"]]

        self.spine = [c for c in channels if "Spine" in c["name"] or "Chest" in c["name"]]
        self.tail = [c for c in channels if "Tail_" in c["name"]]

        self.extra = [
            c for c in channels
            if c not in self.left_leg
            and c not in self.right_leg
            and c not in self.spine
            and c not in self.tail
        ]

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
