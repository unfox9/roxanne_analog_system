import numpy as np


class HopfOscillator:
    def __init__(self, alpha=20.0, mu=1.0, omega=2.0, dt=0.02):
        """
        alpha: 收斂速度
        mu:    控制振幅 (通常=1)
        omega: 基本角頻率（步頻）
        dt:    積分步長（建議跟 Unity FixedUpdate 對齊）
        """
        self.alpha = alpha
        self.mu = mu
        self.omega = omega
        self.dt = dt
        self.x = 1.0
        self.y = 0.0

    def reset(self, phase: float = 0.0):
        self.x = np.cos(phase)
        self.y = np.sin(phase)

    @property
    def phase(self) -> float:
        return float(np.arctan2(self.y, self.x))

    def step(self, omega_mod: float = 0.0) -> float:
        """
        進一步積分一次，回傳主要輸出 wave（x）。
        omega_mod: 額外的頻率調整（由 SAC / 控制器給）
        """
        omega = self.omega + omega_mod
        r2 = self.x * self.x + self.y * self.y

        dx = self.alpha * (self.mu - r2) * self.x - omega * self.y
        dy = self.alpha * (self.mu - r2) * self.y + omega * self.x

        self.x += dx * self.dt
        self.y += dy * self.dt

        return float(self.x)


class BipedCPG:
    """
    雙足 CPG：
    - left:  左腿
    - right: 右腿（相位反轉）
    同時輸出 spine_amp / tail_amp，供 decoder 使用。
    """

    def __init__(self, dt: float = 0.02, base_omega: float = 2.0):
        self.dt = dt
        self.left = HopfOscillator(omega=base_omega, dt=dt)
        self.right = HopfOscillator(omega=base_omega, dt=dt)

    def reset(self):
        self.left.reset(0.0)
        self.right.reset(np.pi)  # 右腳相位 = 左腳 + π（交替步態）

    def step(self, modulation: np.ndarray) -> dict:
        """
        modulation: shape (8,)
            0: freq scale
            1: global amplitude（未來可用在 decoder）
            2: spine amplitude
            3: tail amplitude
            4: turn（左右轉）
            5–7: 保留未來用
        回傳 dict，給 decoder / mapper 用。
        """
        modulation = np.asarray(modulation, dtype=np.float32)
        modulation = np.clip(modulation, -1.0, 1.0)

        freq_scale = 1.0 + 0.5 * float(modulation[0])

        amp = 0.5 * (float(modulation[1]) + 1.0)

        spine_amp = amp * (0.3 + 0.7 * (float(modulation[2]) * 0.5 + 0.5))
        tail_amp = amp * (0.3 + 0.7 * (float(modulation[3]) * 0.5 + 0.5))
        turn_input = float(modulation[4])

        # 轉彎時：內側腿慢一點，外側快一點
        turn_inner = 1.0 - 0.3 * abs(turn_input)
        turn_outer = 1.0 + 0.3 * abs(turn_input)

        if turn_input >= 0.0:
            # 右轉 → 左腿是「內側腿」
            l_scale = freq_scale * turn_inner
            r_scale = freq_scale * turn_outer
        else:
            # 左轉 → 右腿是內側
            l_scale = freq_scale * turn_outer
            r_scale = freq_scale * turn_inner

        l_mod = (l_scale - 1.0) * self.left.omega
        r_mod = (r_scale - 1.0) * self.right.omega

        left_raw = self.left.step(omega_mod=l_mod)
        right_raw = self.right.step(omega_mod=r_mod)

        left_wave = amp * left_raw
        right_wave = amp * right_raw

        return {
            "left": left_wave,
            "right": right_wave,
            "left_phase": self.left.phase,
            "right_phase": self.right.phase,
            "spine_amp": spine_amp,
            "tail_amp": tail_amp,
        }
