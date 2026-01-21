from __future__ import annotations
import numpy as np


def wrap_pi(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class HopfOscillator:
    def __init__(
        self,
        dt: float = 0.02,
        alpha: float = 20.0,
        mu: float = 1.0,
        init_phase: float = 0.0,
    ):
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.mu = float(mu)

        self.x = float(np.cos(init_phase))
        self.y = float(np.sin(init_phase))

    def reset(self, phase: float = 0.0, radius: float | None = None):
        r = 1.0 if radius is None else float(radius)
        self.x = float(r * np.cos(phase))
        self.y = float(r * np.sin(phase))

    @property
    def phase(self) -> float:
        return float(np.arctan2(self.y, self.x))

    @property
    def radius(self) -> float:
        return float(np.sqrt(self.x * self.x + self.y * self.y))

    def step(
        self,
        omega: float,
        mu: float | None = None,
        freeze: bool = False,
    ) -> float:
        if freeze:
            return float(self.x)

        omega = float(omega)
        mu_eff = self.mu if mu is None else float(mu)

        r2 = self.x * self.x + self.y * self.y

        dx = self.alpha * (mu_eff - r2) * self.x - omega * self.y
        dy = self.alpha * (mu_eff - r2) * self.y + omega * self.x

        self.x += dx * self.dt
        self.y += dy * self.dt

        return float(self.x)
