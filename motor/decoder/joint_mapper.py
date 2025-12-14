import numpy as np
from motor.schema.action_schema import ActionSchema


def _sin(x: float) -> float:
    return float(np.sin(float(x)))


class JointMapper:
    def __init__(self, schema: ActionSchema):
        self.schema = schema
        self.action_dim = schema.action_dim

        # --- locomotion core ---
        self.leg_l = schema.get("leg_l")
        self.leg_r = schema.get("leg_r")
        self.knee_l = schema.get("knee_l")
        self.knee_r = schema.get("knee_r")
        self.ankle_l = schema.get("ankle_l")
        self.ankle_r = schema.get("ankle_r")
        self.toe_l = schema.get("toe_l")
        self.toe_r = schema.get("toe_r")

        self.spine = schema.get("spine")
        self.tail = schema.get("tail")

        # --- locomotion assistants ---
        self.arm_l = schema.get("arm_l")
        self.arm_r = schema.get("arm_r")
        self.elbow_l = schema.get("elbow_l")
        self.elbow_r = schema.get("elbow_r")
        self.wrist_l = schema.get("wrist_l")
        self.wrist_r = schema.get("wrist_r")

        self.neck = schema.get("neck")
        self.head = schema.get("head")

        self.hip_gain = {"X": 1.00, "Y": 1.00, "Z": 1.00}

        self.knee_gain_x = 1.00
        self.ankle_gain_x = 1.00
        self.toe_gain_x = 1.00

        self.knee_phase_off = +0.6
        self.ankle_phase_off = -0.6
        self.toe_phase_off = -1.0

        self.spine_gain = {"X": 1.00, "Y": 1.00, "Z": 1.00}
        self.tail_gain = {"X": 1.00, "Y": 1.00}

        self.arm_gain = {"X": 1.00, "Y": 1.00, "Z": 1.00}
        self.elbow_gain_x = 1.00
        self.wrist_gain = {"X": 1.00, "Y": 1.00, "Z": 1.00}

        self.neck_gain = {"X": 1.00, "Y": 1.00, "Z": 1.00}
        self.head_gain = {"X": 1.00, "Y": 1.00, "Z": 1.00}

        self.posture_gain = {
            "ankle_x": 0.30,
            "knee_x": 0.18,
            "hip_fb": -0.12,
            "toe_x": 0.10,
        }

        self.roll_gain = {"hip_lr": 0.10}

    def _set_group(
        self,
        out: np.ndarray,
        group,
        value: float,
        axis: str | None = None,
        add: bool = False,
    ):
        if not group:
            return
        for c in group:
            if axis is None or str(c.get("axis", "")).upper() == axis:
                idx = int(c["index"])
                if add:
                    out[idx] += float(value)
                else:
                    out[idx] = float(value)

    def decode(self, cpg_state) -> np.ndarray:
        out = np.zeros(self.action_dim, dtype=np.float32)

        left_wave = float(cpg_state["left"])
        right_wave = float(cpg_state["right"])
        phiL = float(cpg_state["left_phase"])
        phiR = float(cpg_state["right_phase"])

        spine_amp = float(cpg_state.get("spine_amp", 0.0))
        tail_amp = float(cpg_state.get("tail_amp", 0.0))

        posture = float(cpg_state.get("posture", 0.0))
        roll = float(cpg_state.get("roll", 0.0))

        for ax, g in self.hip_gain.items():
            self._set_group(out, self.leg_l, g * left_wave, axis=ax)
            self._set_group(out, self.leg_r, g * right_wave, axis=ax)

        kneeL = self.knee_gain_x * _sin(phiL + self.knee_phase_off)
        kneeR = self.knee_gain_x * _sin(phiR + self.knee_phase_off)
        ankleL = self.ankle_gain_x * _sin(phiL + self.ankle_phase_off)
        ankleR = self.ankle_gain_x * _sin(phiR + self.ankle_phase_off)
        toeL = self.toe_gain_x * _sin(phiL + self.toe_phase_off)
        toeR = self.toe_gain_x * _sin(phiR + self.toe_phase_off)

        self._set_group(out, self.knee_l, kneeL, axis="X")
        self._set_group(out, self.knee_r, kneeR, axis="X")
        self._set_group(out, self.ankle_l, ankleL, axis="X")
        self._set_group(out, self.ankle_r, ankleR, axis="X")
        self._set_group(out, self.toe_l, toeL, axis="X")
        self._set_group(out, self.toe_r, toeR, axis="X")

        if posture != 0.0:
            self._set_group(
                out,
                self.ankle_l,
                self.posture_gain["ankle_x"] * posture,
                axis="X",
                add=True,
            )
            self._set_group(
                out,
                self.ankle_r,
                self.posture_gain["ankle_x"] * posture,
                axis="X",
                add=True,
            )
            self._set_group(
                out,
                self.knee_l,
                self.posture_gain["knee_x"] * posture,
                axis="X",
                add=True,
            )
            self._set_group(
                out,
                self.knee_r,
                self.posture_gain["knee_x"] * posture,
                axis="X",
                add=True,
            )
            self._set_group(
                out,
                self.toe_l,
                self.posture_gain["toe_x"] * posture,
                axis="X",
                add=True,
            )
            self._set_group(
                out,
                self.toe_r,
                self.posture_gain["toe_x"] * posture,
                axis="X",
                add=True,
            )
            self._set_group(
                out,
                self.leg_l,
                self.posture_gain["hip_fb"] * posture,
                axis="Y",
                add=True,
            )
            self._set_group(
                out,
                self.leg_r,
                self.posture_gain["hip_fb"] * posture,
                axis="Y",
                add=True,
            )

        if roll != 0.0:
            g = self.roll_gain["hip_lr"]
            self._set_group(out, self.leg_l, g * roll, axis="X", add=True)
            self._set_group(out, self.leg_r, -g * roll, axis="X", add=True)

        for i, c in enumerate(self.spine):
            ax = str(c.get("axis", "")).upper()
            gain = self.spine_gain.get(ax, 0.0)
            phase = phiL + i * 0.35
            out[int(c["index"])] = float(spine_amp * gain * np.sin(phase))

        for i, c in enumerate(self.tail):
            ax = str(c.get("axis", "")).upper()
            gain = self.tail_gain.get(ax, 0.0)
            phase = phiL + np.pi + i * 0.45
            out[int(c["index"])] = float(tail_amp * gain * np.sin(phase))

        for ax, g in self.arm_gain.items():
            self._set_group(out, self.arm_l, g * right_wave, axis=ax)
            self._set_group(out, self.arm_r, g * left_wave, axis=ax)

        self._set_group(out, self.elbow_l, self.elbow_gain_x * right_wave, axis="X")
        self._set_group(out, self.elbow_r, self.elbow_gain_x * left_wave, axis="X")

        for ax, g in self.wrist_gain.items():
            self._set_group(out, self.wrist_l, g * right_wave, axis=ax)
            self._set_group(out, self.wrist_r, g * left_wave, axis=ax)

        avg = 0.5 * (left_wave + right_wave)
        for ax, g in self.neck_gain.items():
            self._set_group(out, self.neck, -g * avg, axis=ax)
        for ax, g in self.head_gain.items():
            self._set_group(out, self.head, -g * avg, axis=ax)

        np.clip(out, -1.0, 1.0, out=out)
        return out
