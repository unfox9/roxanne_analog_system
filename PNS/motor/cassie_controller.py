from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

import numpy as np
import mujoco


@dataclass(frozen=True)
class JointPDGains:
    """Joint-side PD gains.

    Units:
        kp: N*m / rad
        kd: N*m*s / rad

    These are *joint-side* gains.  ``CassieController`` converts the requested
    joint torque to MuJoCo motor control using each actuator's transmission
    gear ratio.
    """

    kp: float
    kd: float


@dataclass(frozen=True)
class CassieControlOutput:
    """Diagnostics for one controller update."""

    ctrl: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    q_des: np.ndarray
    qd_des: np.ndarray
    position_error: np.ndarray
    joint_torque: np.ndarray
    saturated: np.ndarray

    @property
    def saturation_count(self) -> int:
        return int(np.count_nonzero(self.saturated))


class CassieController:
    """Low-level joint PD controller for Cassie's ten leg motors.

    Pipeline::

        q_des from gait decoder
                |
                v
        joint-side PD torque
                |
                v
        tau_joint / actuator gear
                |
                v
        MuJoCo motor ctrl

    This class intentionally knows nothing about CPGs or gait phase.  It only
    tracks desired joint positions/velocities.

    Notes
    -----
    Cassie's XML uses ``<motor>`` actuators with scalar joint transmissions.
    For this direct motor model the actuator output force equals ``ctrl`` and
    the joint transmission gear maps that force to generalized joint torque.
    Therefore, for these hinge actuators::

        ctrl = tau_joint / gear

    before applying the actuator's ``ctrlrange`` limit.
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

    # Conservative first-pass gains for the rail-constrained experiment.
    # They are not claimed to be an optimal Cassie gain set; the first goal is
    # controlled motion without violent saturation.  Tune after the first
    # MuJoCo run.
    DEFAULT_GAINS: Mapping[str, JointPDGains] = {
        "left-hip-roll": JointPDGains(kp=70.0, kd=6.0),
        "left-hip-yaw": JointPDGains(kp=55.0, kd=5.0),
        "left-hip-pitch": JointPDGains(kp=110.0, kd=9.0),
        "left-knee": JointPDGains(kp=180.0, kd=12.0),
        "left-foot": JointPDGains(kp=70.0, kd=5.0),
        "right-hip-roll": JointPDGains(kp=70.0, kd=6.0),
        "right-hip-yaw": JointPDGains(kp=55.0, kd=5.0),
        "right-hip-pitch": JointPDGains(kp=110.0, kd=9.0),
        "right-knee": JointPDGains(kp=180.0, kd=12.0),
        "right-foot": JointPDGains(kp=70.0, kd=5.0),
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        gains: Mapping[str, JointPDGains] | None = None,
        position_error_limit_rad: float | None = 0.60,
        ctrl_scale: float = 1.0,
    ) -> None:
        """Create a controller and resolve all model indices by name.

        Parameters
        ----------
        model:
            Loaded Cassie MuJoCo model.
        gains:
            Optional per-joint gain map. Missing entries are filled from
            ``DEFAULT_GAINS``.
        position_error_limit_rad:
            Optional safety clamp on the position error *before* multiplying
            by Kp.  ``0.60`` rad is deliberately conservative for the first
            rail test. Set to ``None`` to disable.
        ctrl_scale:
            Global multiplier applied to the computed actuator command before
            ctrlrange clipping. Useful for cautious bring-up (e.g. 0.5).
        """
        if position_error_limit_rad is not None and position_error_limit_rad <= 0.0:
            raise ValueError("position_error_limit_rad must be > 0 or None")
        if not 0.0 < ctrl_scale <= 1.0:
            raise ValueError("ctrl_scale must satisfy 0 < ctrl_scale <= 1")

        self.model = model
        self.position_error_limit_rad = position_error_limit_rad
        self.ctrl_scale = float(ctrl_scale)

        merged_gains = dict(self.DEFAULT_GAINS)
        if gains is not None:
            unknown = set(gains) - set(self.ACTUATOR_ORDER)
            if unknown:
                raise KeyError(f"unknown Cassie joint gain names: {sorted(unknown)}")
            merged_gains.update(gains)
        self.gains = merged_gains

        self._actuator_ids = np.empty(len(self.ACTUATOR_ORDER), dtype=np.int32)
        self._joint_ids = np.empty(len(self.ACTUATOR_ORDER), dtype=np.int32)
        self._qpos_adr = np.empty(len(self.ACTUATOR_ORDER), dtype=np.int32)
        self._dof_adr = np.empty(len(self.ACTUATOR_ORDER), dtype=np.int32)
        self._gear = np.empty(len(self.ACTUATOR_ORDER), dtype=np.float64)
        self._ctrl_lo = np.empty(len(self.ACTUATOR_ORDER), dtype=np.float64)
        self._ctrl_hi = np.empty(len(self.ACTUATOR_ORDER), dtype=np.float64)
        self._kp = np.empty(len(self.ACTUATOR_ORDER), dtype=np.float64)
        self._kd = np.empty(len(self.ACTUATOR_ORDER), dtype=np.float64)

        for i, name in enumerate(self.ACTUATOR_ORDER):
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

            if actuator_id < 0:
                raise ValueError(f"MuJoCo actuator not found: {name}")
            if joint_id < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")

            self._actuator_ids[i] = actuator_id
            self._joint_ids[i] = joint_id
            self._qpos_adr[i] = int(model.jnt_qposadr[joint_id])
            self._dof_adr[i] = int(model.jnt_dofadr[joint_id])

            # Cassie uses a scalar joint transmission, so gear[0] is the
            # relevant transmission ratio.
            gear = float(model.actuator_gear[actuator_id, 0])
            if abs(gear) < 1e-12:
                raise ValueError(f"actuator {name!r} has zero transmission gear")
            self._gear[i] = gear

            lo, hi = model.actuator_ctrlrange[actuator_id]
            self._ctrl_lo[i] = float(lo)
            self._ctrl_hi[i] = float(hi)

            gain = self.gains[name]
            if gain.kp < 0.0 or gain.kd < 0.0:
                raise ValueError(f"PD gains must be non-negative for {name}")
            self._kp[i] = float(gain.kp)
            self._kd[i] = float(gain.kd)

    def _targets_to_array(self, targets: Any) -> np.ndarray:
        """Normalize decoder output, mappings, or sequences to actuator order."""
        if hasattr(targets, "as_array"):
            arr = np.asarray(targets.as_array(), dtype=np.float64)
        elif isinstance(targets, Mapping):
            missing = [name for name in self.ACTUATOR_ORDER if name not in targets]
            if missing:
                raise KeyError(f"missing desired positions for: {missing}")
            arr = np.asarray([targets[name] for name in self.ACTUATOR_ORDER], dtype=np.float64)
        else:
            arr = np.asarray(targets, dtype=np.float64)

        if arr.shape != (len(self.ACTUATOR_ORDER),):
            raise ValueError(
                f"expected {len(self.ACTUATOR_ORDER)} desired positions, got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("desired joint positions contain NaN or infinity")
        return arr

    def _velocities_to_array(self, qd_des: Any | None) -> np.ndarray:
        if qd_des is None:
            return np.zeros(len(self.ACTUATOR_ORDER), dtype=np.float64)

        if isinstance(qd_des, Mapping):
            missing = [name for name in self.ACTUATOR_ORDER if name not in qd_des]
            if missing:
                raise KeyError(f"missing desired velocities for: {missing}")
            arr = np.asarray([qd_des[name] for name in self.ACTUATOR_ORDER], dtype=np.float64)
        else:
            arr = np.asarray(qd_des, dtype=np.float64)

        if arr.shape != (len(self.ACTUATOR_ORDER),):
            raise ValueError(
                f"expected {len(self.ACTUATOR_ORDER)} desired velocities, got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("desired joint velocities contain NaN or infinity")
        return arr

    def read_state(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Read the ten actuated joint positions and velocities by resolved address."""
        q = np.asarray(data.qpos[self._qpos_adr], dtype=np.float64).copy()
        qd = np.asarray(data.qvel[self._dof_adr], dtype=np.float64).copy()
        return q, qd

    def compute(
        self,
        data: mujoco.MjData,
        q_des: Any,
        qd_des: Any | None = None,
    ) -> CassieControlOutput:
        """Compute motor controls without mutating ``data.ctrl``."""
        q_target = self._targets_to_array(q_des)
        qd_target = self._velocities_to_array(qd_des)
        q, qd = self.read_state(data)

        error = q_target - q
        pd_error = error.copy()
        if self.position_error_limit_rad is not None:
            lim = float(self.position_error_limit_rad)
            pd_error = np.clip(pd_error, -lim, lim)

        # Joint-side PD law.
        tau_joint = self._kp * pd_error + self._kd * (qd_target - qd)

        # MuJoCo <motor>: actuator force == ctrl.  The scalar joint
        # transmission gear maps actuator force to generalized joint torque.
        raw_ctrl = (tau_joint / self._gear) * self.ctrl_scale
        ctrl = np.clip(raw_ctrl, self._ctrl_lo, self._ctrl_hi)
        saturated = np.abs(ctrl - raw_ctrl) > 1e-12

        return CassieControlOutput(
            ctrl=ctrl,
            q=q,
            qd=qd,
            q_des=q_target,
            qd_des=qd_target,
            position_error=error,
            joint_torque=tau_joint,
            saturated=saturated,
        )

    def apply(
        self,
        data: mujoco.MjData,
        q_des: Any,
        qd_des: Any | None = None,
    ) -> CassieControlOutput:
        """Compute controls and write them into Cassie's ten ``data.ctrl`` slots."""
        out = self.compute(data, q_des=q_des, qd_des=qd_des)
        data.ctrl[self._actuator_ids] = out.ctrl
        return out

    @property
    def actuator_ids(self) -> np.ndarray:
        return self._actuator_ids.copy()

    @property
    def gears(self) -> np.ndarray:
        return self._gear.copy()

    @property
    def ctrl_ranges(self) -> np.ndarray:
        return np.column_stack((self._ctrl_lo, self._ctrl_hi))


if __name__ == "__main__":
    print(
        "cassie_controller.py defines CassieController. "
        "Load a MuJoCo Cassie model and instantiate CassieController(model)."
    )