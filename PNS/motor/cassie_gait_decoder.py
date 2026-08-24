from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math

import numpy as np


class CPGOutputLike(Protocol):
    """Minimal interface expected from the rhythm generator.

    ``IzhikevichCPGOutput`` already satisfies this protocol.  Keeping the
    decoder dependent on a protocol rather than a concrete CPG class lets the
    same decoder be used later with Hopf, Matsuoka, or another oscillator.
    """

    left: float
    right: float


@dataclass(frozen=True)
class LegPattern:
    """How one signed CPG primitive shapes one Cassie leg.

    For a primitive m = flexor - extensor:

        swing  = max(m, 0)
        stance = max(-m, 0)

    Each joint target is then formed from a neutral posture plus separate
    swing/stance excursions.  This is intentionally simple and interpretable:
    the CPG supplies rhythm, while this object supplies pattern formation.
    """

    # Neutral / standing-ish pose [rad].  These values are taken from the
    # Cassie model's supplied ``home`` keyframe for the actuated joints.
    hip_roll_center: float = 0.0
    hip_yaw_center: float = 0.0
    hip_pitch_center: float = 0.497301
    knee_center: float = -1.19970
    foot_center: float = -1.59681

    # Swing (flexor-dominant) excursion [rad].
    # Positive hip pitch is used as the first-pass "leg forward" convention.
    # Knee becomes more negative during swing = more knee flexion.
    hip_pitch_swing: float = 0.22
    knee_swing: float = -0.34
    foot_swing: float = 0.18

    # Stance (extensor-dominant) excursion [rad].
    hip_pitch_stance: float = -0.18
    knee_stance: float = 0.16
    foot_stance: float = -0.08

    # We intentionally keep frontal/transverse motion off for the rail test.
    hip_roll_swing: float = 0.0
    hip_roll_stance: float = 0.0
    hip_yaw_swing: float = 0.0
    hip_yaw_stance: float = 0.0


@dataclass(frozen=True)
class CassieGaitTargets:
    """Desired positions for Cassie's ten actuated leg joints, in radians."""

    left_hip_roll: float
    left_hip_yaw: float
    left_hip_pitch: float
    left_knee: float
    left_foot: float

    right_hip_roll: float
    right_hip_yaw: float
    right_hip_pitch: float
    right_knee: float
    right_foot: float

    def as_array(self) -> np.ndarray:
        """Return targets in the Cassie XML actuator order."""
        return np.asarray(
            [
                self.left_hip_roll,
                self.left_hip_yaw,
                self.left_hip_pitch,
                self.left_knee,
                self.left_foot,
                self.right_hip_roll,
                self.right_hip_yaw,
                self.right_hip_pitch,
                self.right_knee,
                self.right_foot,
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, float]:
        """Return targets keyed by MuJoCo joint/actuator name."""
        return {
            "left-hip-roll": self.left_hip_roll,
            "left-hip-yaw": self.left_hip_yaw,
            "left-hip-pitch": self.left_hip_pitch,
            "left-knee": self.left_knee,
            "left-foot": self.left_foot,
            "right-hip-roll": self.right_hip_roll,
            "right-hip-yaw": self.right_hip_yaw,
            "right-hip-pitch": self.right_hip_pitch,
            "right-knee": self.right_knee,
            "right-foot": self.right_foot,
        }


class CassieGaitDecoder:
    """Translate bilateral CPG activity into Cassie joint-position targets.

    First version philosophy
    ------------------------
    * No balance control.
    * No torque/PD control here.
    * No MuJoCo dependency here.
    * Hip roll and hip yaw stay neutral for the rail-constrained experiment.
    * Hip pitch, knee, and foot receive a simple swing/stance pattern.

    The decoder consumes the Matsuoka-like signed primitive already provided by
    ``IzhikevichCPG``::

        m = flexor_activity - extensor_activity

    Positive m is treated as swing/flexor phase and negative m as
    stance/extensor phase.  Half-wave decomposition gives two non-negative
    control channels::

        swing  = max(m, 0)
        stance = max(-m, 0)

    This keeps the neural interpretation visible instead of hiding it inside a
    sinusoidal lookup table.
    """

    ACTUATOR_ORDER = (
        "left-hip-roll",
        "left-hip-yaw",
        "left-hip-pitch",
        "left-knee",
        "left-foot",
        "right-hip-roll",
        "right-hip-yaw",
        "right-hip-pitch",
        "right-knee",
        "right-foot",
    )

    # Cassie XML joint limits.  XML angles are degrees; MuJoCo runtime qpos is
    # radians.  A small margin keeps the first experiment away from hard stops.
    _RAW_LIMITS_DEG = {
        "left-hip-roll": (-15.0, 22.5),
        "left-hip-yaw": (-22.5, 22.5),
        "left-hip-pitch": (-50.0, 80.0),
        "left-knee": (-164.0, -37.0),
        "left-foot": (-140.0, -30.0),
        "right-hip-roll": (-22.5, 15.0),
        "right-hip-yaw": (-22.5, 22.5),
        "right-hip-pitch": (-50.0, 80.0),
        "right-knee": (-164.0, -37.0),
        "right-foot": (-140.0, -30.0),
    }

    def __init__(
        self,
        left: LegPattern | None = None,
        right: LegPattern | None = None,
        amplitude_scale: float = 1.0,
        joint_limit_margin_deg: float = 3.0,
        deadband: float = 0.02,
        dt: float = 0.002,
        primitive_tau_s: float = 0.060,
    ) -> None:
        if amplitude_scale < 0.0:
            raise ValueError("amplitude_scale must be >= 0")
        if joint_limit_margin_deg < 0.0:
            raise ValueError("joint_limit_margin_deg must be >= 0")
        if not 0.0 <= deadband < 1.0:
            raise ValueError("deadband must satisfy 0 <= deadband < 1")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if primitive_tau_s < 0.0:
            raise ValueError("primitive_tau_s must be >= 0")

        self.left = left or LegPattern()
        self.right = right or LegPattern()
        self.amplitude_scale = float(amplitude_scale)
        self.deadband = float(deadband)
        self.dt = float(dt)
        self.primitive_tau_s = float(primitive_tau_s)
        self._primitive_alpha = (
            1.0
            if self.primitive_tau_s == 0.0
            else 1.0 - math.exp(-self.dt / self.primitive_tau_s)
        )
        self._left_primitive = 0.0
        self._right_primitive = 0.0

        margin = math.radians(float(joint_limit_margin_deg))
        self._limits: dict[str, tuple[float, float]] = {}
        for name, (lo_deg, hi_deg) in self._RAW_LIMITS_DEG.items():
            lo = math.radians(lo_deg) + margin
            hi = math.radians(hi_deg) - margin
            if lo >= hi:
                raise ValueError(f"joint-limit margin is too large for {name}")
            self._limits[name] = (lo, hi)

    def reset(self, left_primitive: float = 0.0, right_primitive: float = 0.0) -> None:
        """Reset the small muscle/pattern-formation filter state."""
        self._left_primitive = self._clip_unit(left_primitive)
        self._right_primitive = self._clip_unit(right_primitive)

    def _filter_primitives(self, left: float, right: float) -> tuple[float, float]:
        """Low-pass neural primitives before turning them into joint motion.

        Population firing can switch rapidly even when spike rates themselves
        are filtered.  Real muscle activation and motor drives are not
        instantaneous, so this tiny first-order filter prevents target-angle
        discontinuities without changing the CPG rhythm.
        """
        a = self._primitive_alpha
        self._left_primitive += a * (left - self._left_primitive)
        self._right_primitive += a * (right - self._right_primitive)
        return self._left_primitive, self._right_primitive

    @staticmethod
    def _clip_unit(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    def _apply_deadband(self, x: float) -> float:
        """Remove tiny CPG chatter while preserving the [-1, 1] range."""
        x = self._clip_unit(x)
        a = abs(x)
        if a <= self.deadband:
            return 0.0
        # Remap [deadband, 1] -> [0, 1] continuously.
        y = (a - self.deadband) / (1.0 - self.deadband)
        return math.copysign(y, x)

    def _clip_joint(self, name: str, value: float) -> float:
        lo, hi = self._limits[name]
        return max(lo, min(hi, float(value)))

    def _decode_leg(
        self,
        side: str,
        primitive: float,
        pattern: LegPattern,
    ) -> dict[str, float]:
        m = self._apply_deadband(primitive)
        swing = max(m, 0.0)
        stance = max(-m, 0.0)
        s = self.amplitude_scale

        targets = {
            f"{side}-hip-roll": (
                pattern.hip_roll_center
                + s * (pattern.hip_roll_swing * swing + pattern.hip_roll_stance * stance)
            ),
            f"{side}-hip-yaw": (
                pattern.hip_yaw_center
                + s * (pattern.hip_yaw_swing * swing + pattern.hip_yaw_stance * stance)
            ),
            f"{side}-hip-pitch": (
                pattern.hip_pitch_center
                + s * (pattern.hip_pitch_swing * swing + pattern.hip_pitch_stance * stance)
            ),
            f"{side}-knee": (
                pattern.knee_center
                + s * (pattern.knee_swing * swing + pattern.knee_stance * stance)
            ),
            f"{side}-foot": (
                pattern.foot_center
                + s * (pattern.foot_swing * swing + pattern.foot_stance * stance)
            ),
        }

        return {name: self._clip_joint(name, q) for name, q in targets.items()}

    def decode_primitives(self, left: float, right: float) -> CassieGaitTargets:
        """Decode two signed leg primitives directly.

        This is also the compatibility seam for future Hopf/Matsuoka baselines:
        any oscillator that can provide one signed value per leg can reuse the
        exact same Cassie pattern-formation layer.
        """
        left_m = self._clip_unit(left)
        right_m = self._clip_unit(right)
        left_m, right_m = self._filter_primitives(left_m, right_m)

        left_targets = self._decode_leg("left", left_m, self.left)
        right_targets = self._decode_leg("right", right_m, self.right)

        return CassieGaitTargets(
            left_hip_roll=left_targets["left-hip-roll"],
            left_hip_yaw=left_targets["left-hip-yaw"],
            left_hip_pitch=left_targets["left-hip-pitch"],
            left_knee=left_targets["left-knee"],
            left_foot=left_targets["left-foot"],
            right_hip_roll=right_targets["right-hip-roll"],
            right_hip_yaw=right_targets["right-hip-yaw"],
            right_hip_pitch=right_targets["right-hip-pitch"],
            right_knee=right_targets["right-knee"],
            right_foot=right_targets["right-foot"],
        )

    def decode(self, cpg: CPGOutputLike) -> CassieGaitTargets:
        """Decode one CPG output object into ten desired joint positions."""
        return self.decode_primitives(cpg.left, cpg.right)


if __name__ == "__main__":
    # Dependency-free smoke test: pretend the left leg is in flexor/swing phase
    # and the right leg is in extensor/stance phase.
    @dataclass(frozen=True)
    class DummyCPGOutput:
        left: float
        right: float
        left_flexor: float = 0.8
        left_extensor: float = 0.2
        right_flexor: float = 0.2
        right_extensor: float = 0.8

    decoder = CassieGaitDecoder()
    dummy = DummyCPGOutput(left=0.6, right=-0.6)
    # Let the 60 ms primitive filter settle before printing.
    for _ in range(100):
        targets = decoder.decode(dummy)

    print("Cassie gait decoder smoke test")
    for name, value in targets.as_dict().items():
        print(f"{name:>17s}: {value:+.4f} rad ({math.degrees(value):+7.2f} deg)")