from abc import ABC, abstractmethod
import numpy as np

class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = dtype

        self.low = np.full(self.shape, low, dtype=dtype)
        self.high = np.full(self.shape, high, dtype=dtype)

    def sample(self):
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