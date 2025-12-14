from __future__ import annotations
import numpy as np

from motor.cpg.hopf import HopfOscillator, wrap_pi
from motor.cpg.wave_shaper import sigmoid01, clip, duty_wave


class BalanceReflex:
    def __init__(self, kp_pitch=1.0, kd_pitch=0.2, kp_roll=1.0, kd_roll=0.2, clip=1.0):
        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.clip = clip

    def __call__(self, reflex: np.ndarray, gain: float):
        pitch = float(reflex[0])
        roll = float(reflex[1])
        pitch_rate = float(reflex[2])
        roll_rate = float(reflex[3])

        posture = -(self.kp_pitch * pitch + self.kd_pitch * pitch_rate)
        rollcmd = -(self.kp_roll * roll + self.kd_roll * roll_rate)

        posture = np.clip(gain * posture, -self.clip, self.clip)
        rollcmd = np.clip(gain * rollcmd, -self.clip, self.clip)
        return float(posture), float(rollcmd)


class BipedCPG:
    def __init__(
        self,
        dt: float = 0.02,
        base_omega: float = 2.0,
        alpha: float = 20.0,
        mu: float = 1.0,
        turn_strength: float = 0.35,
        phase_lock_k: float = 2.0,
        freeze_amp_threshold: float = 0.02,
        enable_phase_lock: bool = True,
        param_dim: int = 8,
    ):
        self.dt = float(dt)
        self.base_omega = float(base_omega)
        self.turn_strength = float(turn_strength)

        self.phase_lock_k = float(phase_lock_k)
        self.freeze_amp_threshold = float(freeze_amp_threshold)
        self.enable_phase_lock = bool(enable_phase_lock)

        self.left = HopfOscillator(dt=dt, alpha=alpha, mu=mu, init_phase=0.0)
        self.right = HopfOscillator(dt=dt, alpha=alpha, mu=mu, init_phase=np.pi)

        self.param_dim = int(param_dim)

        self.balance = BalanceReflex()

    def reset(self):
        self.left.reset(phase=0.0)
        self.right.reset(phase=np.pi)

    def step(self, modulation: np.ndarray, reflex: np.ndarray | None = None) -> dict:
        m = np.asarray(modulation, dtype=np.float32).reshape(-1)
        if m.shape[0] < self.param_dim:
            raise ValueError(
                f"modulation must have dim {self.param_dim}, got {m.shape[0]}"
            )
        m = np.clip(m, -1.0, 1.0)

        freq_scale = 1.0 + 0.5 * float(m[0])
        amp = sigmoid01(float(m[1]))

        duty = 0.3 + 0.4 * sigmoid01(float(m[2]))
        sharp = sigmoid01(float(m[3]))
        turn = clip(float(m[4]), -1.0, 1.0)

        spine_amp = amp * sigmoid01(float(m[5]))
        tail_amp = amp * sigmoid01(float(m[6]))

        omega = self.base_omega * freq_scale
        domega = self.turn_strength * turn * omega
        omega_L = max(0.0, omega - domega)
        omega_R = max(0.0, omega + domega)

        freeze = amp < self.freeze_amp_threshold

        _ = self.left.step(omega=omega_L, freeze=freeze)
        _ = self.right.step(omega=omega_R, freeze=freeze)

        if self.enable_phase_lock and (not freeze):
            phiL = self.left.phase
            phiR = self.right.phase
            err = wrap_pi((phiR - phiL) - np.pi)
            self.left.reset(
                phase=phiL + 0.5 * self.phase_lock_k * err * self.dt,
                radius=self.left.radius,
            )
            self.right.reset(
                phase=phiR - 0.5 * self.phase_lock_k * err * self.dt,
                radius=self.right.radius,
            )

        phiL = self.left.phase
        phiR = self.right.phase

        left_raw = duty_wave(phiL, duty=duty, sharp=sharp)
        right_raw = duty_wave(phiR, duty=duty, sharp=sharp)

        left = float(amp * left_raw)
        right = float(amp * right_raw)

        balance_gain = 0.5 * (float(m[7]) + 1.0)
        posture = 0.0
        rollcmd = 0.0
        if reflex is not None and len(reflex) >= 4:
            posture, rollcmd = self.balance(reflex, gain=balance_gain)

        return {
            "left": left,
            "right": right,
            "left_phase": float(phiL),
            "right_phase": float(phiR),
            "spine_amp": float(spine_amp),
            "tail_amp": float(tail_amp),
            "balance_gain": balance_gain,
            "left_sin": float(np.sin(phiL)),
            "left_cos": float(np.cos(phiL)),
            "right_sin": float(np.sin(phiR)),
            "right_cos": float(np.cos(phiR)),
            "duty": float(duty),
            "sharp": float(sharp),
            "amp": float(amp),
            "freq_scale": float(freq_scale),
            "turn": float(turn),
            "balance_gain": float(balance_gain),
            "posture": posture,
            "rollcmd": rollcmd,
        }
