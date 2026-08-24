from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

import numpy as np
import mujoco


@dataclass(frozen=True)
class StandingPose:
    """Nominal Cassie standing pose for the ten actuated joints [rad].

    These values match the standing-ish center used by the gait decoder and
    the supplied Cassie home keyframe closely enough for first bring-up.
    """

    left_hip_roll: float = 0.00449956
    left_hip_yaw: float = 0.0
    left_hip_pitch: float = 0.497301
    left_knee: float = -1.19970
    left_foot: float = -1.59681

    right_hip_roll: float = -0.00449956
    right_hip_yaw: float = 0.0
    right_hip_pitch: float = 0.497301
    right_knee: float = -1.19970
    right_foot: float = -1.59681

    def as_dict(self) -> dict[str, float]:
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


@dataclass(frozen=True)
class StandingReflexConfig:
    """Simple hand-coded posture reflexes.

    Version 0 deliberately focuses on vertical support.  With the pelvis pitch
    joint locked in the XML, the height reflex is enough to answer the first
    question: can the legs support the body instead of collapsing?

    The pitch fields are present for the next experiment, but are ZERO by
    default because the correct sign/magnitude depends on Cassie's joint and
    contact convention.  Enable them only after x+z standing works.
    """

    # Pelvis rail coordinates are relative to the body's XML pose, so home is
    # z = 0 and pitch = 0 for the x+z(+pitch) rail models.
    target_z: float = 0.0
    target_pitch: float = 0.0

    # Height -> knee extension reflex.
    # If the pelvis drops, z becomes negative, therefore target_z - z > 0.
    # Adding a positive correction makes Cassie's knee angle less negative,
    # which is the extension direction used by our gait decoder.
    height_kp_knee: float = 1.8      # rad knee / m pelvis error
    height_kd_knee: float = 0.20     # rad knee / (m/s)
    height_ki_knee: float = 0.30     # rad knee / (m*s)
    max_height_integral: float = 0.10  # m*s
    max_height_knee_delta: float = 0.30  # rad

    # Optional sagittal pitch reflexes.  Keep at zero for the first x+z test.
    # The signs are intentionally exposed instead of hidden in the code.
    pitch_kp_hip: float = 1.0
    pitch_kd_hip: float = 0.08
    pitch_kp_foot: float = 0.40
    pitch_kd_foot: float = 0.06
    max_pitch_hip_delta: float = 0.35
    max_pitch_foot_delta: float = 0.30

    # Move from the current joint configuration toward the standing target
    # smoothly.  This avoids an instantaneous position-command jump at reset.
    ramp_time_s: float = 1.0

    # Leave a little room before mechanical joint stops.
    joint_limit_margin_deg: float = 2.0


@dataclass(frozen=True)
class CassieStandingOutput:
    """Standing target plus diagnostics for one update."""

    targets: Mapping[str, float]
    z: float | None
    z_velocity: float | None
    z_error: float | None
    pitch: float | None
    pitch_rate: float | None
    pitch_error: float | None
    knee_height_delta: float
    hip_pitch_delta: float
    foot_pitch_delta: float
    ramp: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.targets)


class CassieStanding:
    """Small standing/posture layer for rail-constrained Cassie.

    This is *not* a learned controller.  It is an intentionally interpretable
    reflex controller that generates desired joint positions for the existing
    ``CassieController`` PD layer.

    Data flow::

        pelvis z / pitch feedback
                 +
          nominal standing pose
                 |
                 v
          10 joint q_des targets
                 |
                 v
          CassieController (PD)

    Recommended bring-up sequence
    -----------------------------
    1. XML: pelvis x + z are free, pitch is locked.
    2. CPG: OFF.
    3. CassieStanding: ON.
    4. Verify the body can support itself for 5-10 s.
    5. Only then re-enable pelvis pitch and tune the pitch reflex.
    6. Only after standing works should CPG gait offsets be added on top.
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

    def __init__(
        self,
        model: mujoco.MjModel,
        pose: StandingPose | None = None,
        reflex: StandingReflexConfig | None = None,
        z_joint_name: str = "pelvis-rail-z",
        pitch_joint_name: str = "pelvis-rail-pitch",
    ) -> None:
        self.model = model
        self.pose = pose or StandingPose()
        self.reflex = reflex or StandingReflexConfig()
        self.z_joint_name = z_joint_name
        self.pitch_joint_name = pitch_joint_name

        if self.reflex.ramp_time_s < 0.0:
            raise ValueError("ramp_time_s must be >= 0")
        if self.reflex.max_height_knee_delta < 0.0:
            raise ValueError("max_height_knee_delta must be >= 0")
        if self.reflex.max_pitch_hip_delta < 0.0:
            raise ValueError("max_pitch_hip_delta must be >= 0")
        if self.reflex.max_pitch_foot_delta < 0.0:
            raise ValueError("max_pitch_foot_delta must be >= 0")

        self._joint_ids: dict[str, int] = {}
        self._qpos_adr: dict[str, int] = {}
        self._limits: dict[str, tuple[float, float]] = {}

        margin = math.radians(self.reflex.joint_limit_margin_deg)
        for name in self.ACTUATOR_ORDER:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")

            self._joint_ids[name] = joint_id
            self._qpos_adr[name] = int(model.jnt_qposadr[joint_id])

            if bool(model.jnt_limited[joint_id]):
                lo, hi = model.jnt_range[joint_id]
                lo = float(lo) + margin
                hi = float(hi) - margin
                if lo >= hi:
                    raise ValueError(f"joint limit margin too large for {name}")
                self._limits[name] = (lo, hi)
            else:
                self._limits[name] = (-math.inf, math.inf)

        self._z_qpos_adr, self._z_dof_adr = self._resolve_optional_joint(z_joint_name)
        self._pitch_qpos_adr, self._pitch_dof_adr = self._resolve_optional_joint(
            pitch_joint_name
        )

        self._elapsed_s = 0.0
        self._height_integral = 0.0
        self._start_pose = self.pose.as_dict()

    def _resolve_optional_joint(self, name: str) -> tuple[int | None, int | None]:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            return None, None
        return int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])

    def _read_actuated_pose(self, data: mujoco.MjData) -> dict[str, float]:
        return {
            name: float(data.qpos[self._qpos_adr[name]])
            for name in self.ACTUATOR_ORDER
        }

    @staticmethod
    def _smoothstep01(x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def _clip_joint(self, name: str, value: float) -> float:
        lo, hi = self._limits[name]
        return max(lo, min(hi, float(value)))

    def reset(self, data: mujoco.MjData) -> None:
        """Start a new standing attempt from the current actuator pose."""
        self._elapsed_s = 0.0
        self._height_integral = 0.0
        self._start_pose = self._read_actuated_pose(data)

    def step(self, data: mujoco.MjData, dt: float) -> CassieStandingOutput:
        """Generate standing joint targets from current posture feedback."""
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        self._elapsed_s += float(dt)
        if self.reflex.ramp_time_s == 0.0:
            ramp = 1.0
        else:
            ramp = self._smoothstep01(self._elapsed_s / self.reflex.ramp_time_s)

        nominal = self.pose.as_dict()
        targets = {
            name: (1.0 - ramp) * self._start_pose[name] + ramp * nominal[name]
            for name in self.ACTUATOR_ORDER
        }


        # ---------------------------------------------------------------
        # Optional pitch reflex.  Gains default to zero for the first test.
        # Positive/negative gain signs are deliberately tunable because the
        # required correction depends on Cassie's coordinate convention.
        # ---------------------------------------------------------------
        pitch = pitch_rate = pitch_error = None
        hip_pitch_delta = 0.0
        foot_pitch_delta = 0.0
        if self._pitch_qpos_adr is not None and self._pitch_dof_adr is not None:
            pitch = float(data.qpos[self._pitch_qpos_adr])
            pitch_rate = float(data.qvel[self._pitch_dof_adr])
            pitch_error = float(self.reflex.target_pitch - pitch)

            hip_pitch_delta = (
                self.reflex.pitch_kp_hip * pitch_error
                - self.reflex.pitch_kd_hip * pitch_rate
            )
            foot_pitch_delta = (
                self.reflex.pitch_kp_foot * pitch_error
                - self.reflex.pitch_kd_foot * pitch_rate
            )

            hip_pitch_delta = float(
                np.clip(
                    hip_pitch_delta,
                    -self.reflex.max_pitch_hip_delta,
                    +self.reflex.max_pitch_hip_delta,
                )
            )
            foot_pitch_delta = float(
                np.clip(
                    foot_pitch_delta,
                    -self.reflex.max_pitch_foot_delta,
                    +self.reflex.max_pitch_foot_delta,
                )
            )

            targets["left-hip-pitch"] += ramp * hip_pitch_delta
            targets["right-hip-pitch"] += ramp * hip_pitch_delta
            targets["left-foot"] += ramp * foot_pitch_delta
            targets["right-foot"] += ramp * foot_pitch_delta


        # ---------------------------------------------------------------
        # Vertical support reflex: pelvis drops -> extend both knees.
        # ---------------------------------------------------------------
        z = z_velocity = z_error = None
        knee_height_delta = 0.0
        if self._z_qpos_adr is not None and self._z_dof_adr is not None:
            z = float(data.qpos[self._z_qpos_adr])
            z_velocity = float(data.qvel[self._z_dof_adr])
            z_error = float(self.reflex.target_z - z)
            # Forward Euler integration
            if abs(pitch or 0.0) < math.radians(10.0):
                self._height_integral += z_error * dt

            # Anti-windup
            self._height_integral = float(
                np.clip(
                    self._height_integral,
                    -self.reflex.max_height_integral,
                    +self.reflex.max_height_integral,
                )
            )

            knee_height_delta = (
                self.reflex.height_kp_knee * z_error
                + self.reflex.height_ki_knee * self._height_integral
                - self.reflex.height_kd_knee * z_velocity
            )
            knee_height_delta = float(
                np.clip(
                    knee_height_delta,
                    -self.reflex.max_height_knee_delta,
                    +self.reflex.max_height_knee_delta,
                )
            )

            # Ramp the reflex in too, so the first simulation step remains tame.
            targets["left-knee"] += knee_height_delta
            targets["right-knee"] += knee_height_delta


        targets = {
            name: self._clip_joint(name, value)
            for name, value in targets.items()
        }

        return CassieStandingOutput(
            targets=targets,
            z=z,
            z_velocity=z_velocity,
            z_error=z_error,
            pitch=pitch,
            pitch_rate=pitch_rate,
            pitch_error=pitch_error,
            knee_height_delta=knee_height_delta,
            hip_pitch_delta=hip_pitch_delta,
            foot_pitch_delta=foot_pitch_delta,
            ramp=ramp,
        )


if __name__ == "__main__":
    print(
        "cassie_standing.py defines CassieStanding. "
        "Use it with a loaded MuJoCo Cassie model and the existing CassieController."
    )