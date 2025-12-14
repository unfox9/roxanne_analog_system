from __future__ import annotations
import numpy as np


def sigmoid01(x: float) -> float:
    return 0.5 * (float(x) + 1.0)


def clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), lo, hi))


def phase01(phase: float) -> float:
    return float((phase % (2.0 * np.pi)) / (2.0 * np.pi))


def tanh_sharpen(y: float, sharp: float) -> float:
    s = clip(sharp, 0.0, 1.0)
    k = 1.0 + 6.0 * s
    return float(np.tanh(k * y) / np.tanh(k))


def duty_wave(phase: float, duty: float, sharp: float) -> float:
    duty = clip(duty, 1e-3, 1.0 - 1e-3)
    u = phase01(phase)

    if u < duty:
        t = u / duty
        y = np.cos(np.pi * t)
    else:
        t = (u - duty) / (1.0 - duty)
        y = -np.cos(np.pi * t)

    return tanh_sharpen(float(y), sharp)


def sin_wave(phase: float, sharp: float = 0.0) -> float:
    y = float(np.sin(phase))
    return tanh_sharpen(y, sharp)


def cos_wave(phase: float, sharp: float = 0.0) -> float:
    y = float(np.cos(phase))
    return tanh_sharpen(y, sharp)
