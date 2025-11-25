from abc import ABC, abstractmethod
import numpy as np

class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        """
        簡易版 gym.spaces.Box：
        - low, high: 可以是 scalar，也可以是 array（你先用 scalar 就好）
        - shape: tuple, 例如 (obs_dim,) 或 (action_dim,)
        """
        self.shape = tuple(shape)
        self.dtype = dtype

        # 全部展開成同 shape 的陣列，方便 .low / .high / .sample 使用
        self.low = np.full(self.shape, low, dtype=dtype)
        self.high = np.full(self.shape, high, dtype=dtype)

    def sample(self):
        # 隨機從 [low, high] 裡抽一個
        return np.random.uniform(self.low, self.high).astype(self.dtype)

class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def close(self):
        pass