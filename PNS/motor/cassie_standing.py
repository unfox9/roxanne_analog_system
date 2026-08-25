from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

import numpy as np
import mujoco


@dataclass(frozen=True)
class StandingPose:
    """Nominal Cassie standing pose for the ten actuated joints [rad]."""

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
    """Hand-coded standing reflexes for the sagittal rail experiment.

    The standing layer now has three deliberately separate feedback channels:

    * height -> bilateral knee extension
    * pelvis pitch -> hip-pitch + foot corrections
    * whole-body COM position relative to the two feet -> hip-pitch + foot corrections

    The final desired joint positions are still tracked by ``CassieController``
    using low-level joint PD.  These gains are first-pass bring-up values, not
    an optimal Cassie controller.
    """

    # Pelvis rail coordinates are relative to the body's XML pose, so home is
    # z = 0 and pitch = 0 for the x+z+pitch rail model.
    target_z: float = 0.0
    target_pitch: float = 0.0

    # Height -> knee extension reflex.
    height_kp_knee: float = 1.8       # rad knee / m pelvis error
    height_kd_knee: float = 0.20      # rad knee / (m/s)
    height_ki_knee: float = 0.30      # rad knee / (m*s)
    max_height_integral: float = 0.10 # m*s
    max_height_knee_delta: float = 0.30

    # Pelvis pitch -> sagittal joint correction.
    # These preserve the tuned values from the uploaded v5 project.
    pitch_kp_hip: float = 1.0
    pitch_kd_hip: float = 0.08
    pitch_kp_foot: float = 0.40
    pitch_kd_foot: float = 0.06
    max_pitch_hip_delta: float = 0.35
    max_pitch_foot_delta: float = 0.30

    # Horizontal COM-over-support feedback.
    #
    # support_error_x is defined as:
    #   (com_x - mean(left_foot_x, right_foot_x)) - reset_reference
    #
    # Therefore support_error_x > 0 means the whole-body COM has moved forward
    # relative to the two feet.  The negative signs in the control law below
    # command the already calibrated recovery direction.
    support_kp_hip: float = 0.60     # rad hip / m relative-x error 
    support_kd_hip: float = 0.08     # rad hip / (m/s) 
    support_kp_foot: float = 0.35    # rad foot / m relative-x error 
    support_kd_foot: float = 0.05    # rad foot / (m/s) 
    max_support_hip_delta: float = 0.18
    max_support_foot_delta: float = 0.15

    # Finite-difference relative-x velocity is low-pass filtered before use.
    # At 2000 Hz physics, 30 ms removes a lot of derivative chatter without
    # making this slow standing reflex feel asleep.
    support_velocity_filter_tau_s: float = 0.030

    # Smooth only the nominal pose transition.  Feedback reflexes remain fully
    # active during the startup ramp so balance is not artificially weakened in
    # the first second.
    ramp_time_s: float = 1.0

    # Height integration is frozen once pitch exceeds this angle because a low
    # pelvis during a fall is no longer a useful "standing height" error.
    height_integral_pitch_guard_deg: float = 10.0

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

    pelvis_x: float
    com_x: float
    support_center_x: float
    pelvis_support_offset_x: float
    com_support_offset_x: float
    support_target_offset_x: float
    support_error_x: float
    support_velocity_x: float

    knee_height_delta: float
    hip_pitch_delta: float
    foot_pitch_delta: float
    hip_support_delta: float
    foot_support_delta: float
    ramp: float

    cop_x: float | None
    ground_normal_force: float
    support_min_x: float | None
    support_max_x: float | None

    def as_dict(self) -> dict[str, float]:
        return dict(self.targets)


class CassieStanding:
    """Standing/posture layer for rail-constrained Cassie.

    This is intentionally interpretable rather than learned.  It generates
    desired joint positions for the existing low-level ``CassieController``.

    Data flow::

        z / pitch / COM-vs-feet x feedback
                       +
                nominal pose
                       |
                       v
              10 joint q_des
                       |
                       v
              joint-level PD
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
        pelvis_body_name: str = "cassie-pelvis",
        left_foot_body_name: str = "left-foot",
        right_foot_body_name: str = "right-foot",
    ) -> None:
        self.model = model
        self.pose = pose or StandingPose()
        self.reflex = reflex or StandingReflexConfig()
        self.z_joint_name = z_joint_name
        self.pitch_joint_name = pitch_joint_name

        if self.reflex.ramp_time_s < 0.0:
            raise ValueError("ramp_time_s must be >= 0")
        if self.reflex.support_velocity_filter_tau_s < 0.0:
            raise ValueError("support_velocity_filter_tau_s must be >= 0")
        if self.reflex.height_integral_pitch_guard_deg < 0.0:
            raise ValueError("height_integral_pitch_guard_deg must be >= 0")

        for field_name in (
            "max_height_knee_delta",
            "max_pitch_hip_delta",
            "max_pitch_foot_delta",
            "max_support_hip_delta",
            "max_support_foot_delta",
            "max_height_integral",
        ):
            if getattr(self.reflex, field_name) < 0.0:
                raise ValueError(f"{field_name} must be >= 0")

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

        self._pelvis_body_id = self._resolve_required_body(pelvis_body_name)
        self._left_foot_body_id = self._resolve_required_body(left_foot_body_name)
        self._right_foot_body_id = self._resolve_required_body(right_foot_body_name)

        self._left_foot_subtree_ids = self._collect_subtree_body_ids(
            self._left_foot_body_id
        )

        self._right_foot_subtree_ids = self._collect_subtree_body_ids(
            self._right_foot_body_id
        )

        self._foot_subtree_ids = (
            self._left_foot_subtree_ids
            | self._right_foot_subtree_ids
        )

        self._elapsed_s = 0.0
        self._height_integral = 0.0
        self._start_pose = self.pose.as_dict()

        # Captured from the actual reset pose rather than assumed to be zero.
        # This preserves Cassie's native COM-to-support geometry even if the foot
        # body origins are not centered under the whole-body COM.
        self._support_target_offset_x = 0.0
        self._support_prev_offset_x: float | None = None
        self._support_velocity_x = 0.0

    def _collect_subtree_body_ids(self, root_body_id: int) -> set[int]:
        body_ids = {root_body_id}

        changed = True
        while changed:
            changed = False

            for body_id in range(self.model.nbody):
                parent_id = int(self.model.body_parentid[body_id])

                if parent_id in body_ids and body_id not in body_ids:
                    body_ids.add(body_id)
                    changed = True

        return body_ids

    def _resolve_optional_joint(self, name: str) -> tuple[int | None, int | None]:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            return None, None
        return int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])

    def _resolve_required_body(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {name}")
        return int(body_id)

    def _read_actuated_pose(self, data: mujoco.MjData) -> dict[str, float]:
        return {
            name: float(data.qpos[self._qpos_adr[name]])
            for name in self.ACTUATOR_ORDER
        }

    def _read_support_geometry(
        self,
        data: mujoco.MjData,
    ) -> tuple[float, float, float, float, float]:
        """Return pelvis x, whole-body COM x, support midpoint, and both offsets.

        ``data.subtree_com[pelvis_body_id]`` is the center of mass of the whole
        MuJoCo body subtree rooted at Cassie's pelvis.  Because the pelvis is the
        root body of the robot, this is the whole-body Cassie COM.

        The two foot *body origins* are still used as the support proxy for this
        experiment.  A later walking controller can replace this midpoint with
        actual contact-aware support or center of pressure.
        """
        pelvis_x = float(data.xpos[self._pelvis_body_id, 0])
        com_x = float(data.subtree_com[self._pelvis_body_id, 0])

        left_foot_x = float(data.xpos[self._left_foot_body_id, 0])
        right_foot_x = float(data.xpos[self._right_foot_body_id, 0])
        support_center_x = 0.5 * (left_foot_x + right_foot_x)

        pelvis_offset_x = pelvis_x - support_center_x
        com_offset_x = com_x - support_center_x

        return (
            pelvis_x,
            com_x,
            support_center_x,
            pelvis_offset_x,
            com_offset_x,
        )

    @staticmethod
    def _smoothstep01(x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def _clip_joint(self, name: str, value: float) -> float:
        lo, hi = self._limits[name]
        return max(lo, min(hi, float(value)))

    def reset(self, data: mujoco.MjData) -> None:
        """Start a new standing attempt from the current MuJoCo pose."""
        self._elapsed_s = 0.0
        self._height_integral = 0.0
        self._start_pose = self._read_actuated_pose(data)

        _, _, _, _, com_offset_x = self._read_support_geometry(data)
        self._support_target_offset_x = com_offset_x
        self._support_prev_offset_x = com_offset_x
        self._support_velocity_x = 0.0

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
        # 1) Pelvis pitch reflex.
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

            # Feedback is deliberately NOT multiplied by the nominal-pose ramp.
            # A balance reflex should not be half-asleep during the first second.
            targets["left-hip-pitch"] += hip_pitch_delta
            targets["right-hip-pitch"] += hip_pitch_delta
            targets["left-foot"] += foot_pitch_delta
            targets["right-foot"] += foot_pitch_delta

        # ---------------------------------------------------------------
        # 2) Horizontal support-centre reflex.
        # ---------------------------------------------------------------
        (
            pelvis_x,
            com_x,
            support_center_x,
            pelvis_support_offset_x,
            com_support_offset_x,
        ) = self._read_support_geometry(data)
        cop_x, ground_normal_force, support_min_x, support_max_x = self._compute_cop_x(data)

        # Positive error means the whole-body COM moved forward relative to the
        # two-foot support midpoint compared with the reset standing geometry.
        support_error_x = com_support_offset_x - self._support_target_offset_x

        if self._support_prev_offset_x is None:
            raw_support_velocity_x = 0.0
        else:
            raw_support_velocity_x = (
                com_support_offset_x - self._support_prev_offset_x
            ) / dt
        self._support_prev_offset_x = com_support_offset_x

        tau = self.reflex.support_velocity_filter_tau_s
        if tau == 0.0:
            self._support_velocity_x = raw_support_velocity_x
        else:
            alpha = dt / (tau + dt)
            self._support_velocity_x += alpha * (
                raw_support_velocity_x - self._support_velocity_x
            )

        # Positive support_error_x means the whole-body COM moved forward
        # relative to the feet.  Negative joint corrections use the already
        # calibrated recovery direction for positive sagittal pitch.
        hip_support_delta = -(
            self.reflex.support_kp_hip * support_error_x
            + self.reflex.support_kd_hip * self._support_velocity_x
        )
        foot_support_delta = -(
            self.reflex.support_kp_foot * support_error_x
            + self.reflex.support_kd_foot * self._support_velocity_x
        )

        hip_support_delta = float(
            np.clip(
                hip_support_delta,
                -self.reflex.max_support_hip_delta,
                +self.reflex.max_support_hip_delta,
            )
        )
        foot_support_delta = float(
            np.clip(
                foot_support_delta,
                -self.reflex.max_support_foot_delta,
                +self.reflex.max_support_foot_delta,
            )
        )

        targets["left-hip-pitch"] += hip_support_delta
        targets["right-hip-pitch"] += hip_support_delta
        targets["left-foot"] += foot_support_delta
        targets["right-foot"] += foot_support_delta

        # ---------------------------------------------------------------
        # 3) Vertical support reflex: pelvis drops -> extend both knees.
        # ---------------------------------------------------------------
        z = z_velocity = z_error = None
        knee_height_delta = 0.0
        if self._z_qpos_adr is not None and self._z_dof_adr is not None:
            z = float(data.qpos[self._z_qpos_adr])
            z_velocity = float(data.qvel[self._z_dof_adr])
            z_error = float(self.reflex.target_z - z)

            # Forward Euler integration.  Freeze it during a large pitch fall.
            pitch_for_guard = pitch if pitch is not None else 0.0
            if abs(pitch_for_guard) < math.radians(
                self.reflex.height_integral_pitch_guard_deg
            ):
                self._height_integral += z_error * dt

            # Simple integral anti-windup clamp.
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
            pelvis_x=pelvis_x,
            com_x=com_x,
            support_center_x=support_center_x,
            pelvis_support_offset_x=pelvis_support_offset_x,
            com_support_offset_x=com_support_offset_x,
            support_target_offset_x=self._support_target_offset_x,
            support_error_x=support_error_x,
            support_velocity_x=self._support_velocity_x,
            knee_height_delta=knee_height_delta,
            hip_pitch_delta=hip_pitch_delta,
            foot_pitch_delta=foot_pitch_delta,
            hip_support_delta=hip_support_delta,
            foot_support_delta=foot_support_delta,
            ramp=ramp,
            cop_x=cop_x,
            ground_normal_force=ground_normal_force,
            support_min_x=support_min_x,
            support_max_x=support_max_x,
        )

    
    def _compute_cop_x(
        self,
        data: mujoco.MjData,
    ) -> tuple[float | None, float, float | None, float | None]:
        """Return sagittal CoP x and total normal contact force."""

        weighted_x = 0.0
        total_normal_force = 0.0
        contact_x_positions = []

        contact_force = np.zeros(6, dtype=float)

        for i in range(data.ncon):
            contact = data.contact[i]

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])

            geom1_is_foot = body1 in self._foot_subtree_ids
            geom2_is_foot = body2 in self._foot_subtree_ids

            # Exactly one side should belong to Cassie's feet.
            if geom1_is_foot == geom2_is_foot:
                continue

            contact_force[:] = 0.0

            mujoco.mj_contactForce(
                self.model,
                data,
                i,
                contact_force,
            )

            normal_force = float(contact_force[0])

            if not np.isfinite(normal_force):
                continue

            if normal_force <= 1e-9:
                continue

            contact_x = float(contact.pos[0])

            if not np.isfinite(contact_x):
                continue

            contact_x_positions.append(contact_x)

            weighted_x += normal_force * contact_x
            total_normal_force += normal_force

        if (
            not contact_x_positions
            or not np.isfinite(total_normal_force)
            or total_normal_force <= 1e-9
        ):
            return None, 0.0, None, None

        cop_x = weighted_x / total_normal_force
        support_min_x = min(contact_x_positions)
        support_max_x = max(contact_x_positions)

        return (
            float(cop_x),
            float(total_normal_force),
            float(support_min_x),
            float(support_max_x),
        )


if __name__ == "__main__":
    print(
        "cassie_standing.py defines CassieStanding. "
        "Use it with a loaded MuJoCo Cassie model and the existing CassieController."
    )