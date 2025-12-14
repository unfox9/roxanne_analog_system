import numpy as np


class ReflexProjector:
    def __init__(self, idx: list[int], clip: float = 5.0):
        self.idx = idx
        self.clip = float(clip)

    @property
    def dim(self) -> int:
        return len(self.idx)

    def __call__(self, obs_1d: np.ndarray) -> np.ndarray:
        x = np.asarray(obs_1d, dtype=np.float32).reshape(-1)
        r = x[self.idx].astype(np.float32)
        np.clip(r, -self.clip, self.clip, out=r)
        return r
