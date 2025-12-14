import numpy as np
import torch

from motor.cpg.biped_cpg import BipedCPG
from motor.decoder.joint_mapper import JointMapper
from motor.schema.action_schema import ActionSchema
from motor.schema.obs_schema import ObsSchema, ObsItem
from motor.decoder.structured_decoder import StructuredDecoder
from motor.reflex.reflex_extractor import ReflexProjector
from motor.residual.basis_projector import BasisProjector


class MotorController:
    def __init__(
        self,
        mapping_path: str,
        dt: float = 0.02,
        latent_dim: int = 8,
        decoder_ckpt: str | None = None,
        residual_scale: float = 0.10,
        device: str | None = None,
        expected_action_dim: int | None = None,
        reflex_idx: list[int] | None = None,
        basis_specs: list[tuple[str, str | None]] | None = None,
        phase_dim: int = 0,
        strict_contiguous_mapping: bool = False,
        reflex_clip: float = 5.0,
    ):
        self.schema = ActionSchema.from_mapping_file(
            mapping_path,
            strict_contiguous=strict_contiguous_mapping,
        )
        if expected_action_dim is not None and self.schema.action_dim != int(
            expected_action_dim
        ):
            raise ValueError(
                f"mapping action_dim({self.schema.action_dim}) != expected_action_dim({expected_action_dim})"
            )

        self.obs_schema = ObsSchema(
            [
                ObsItem("hipY", 1, "reflex"),
                ObsItem("com", 3, "reflex"),
                ObsItem("toes", 16, "reflex"),
            ]
        )

        self.dt = dt
        self.cpg = BipedCPG(dt=dt, param_dim=9)
        cpg_dim = self.cpg.param_dim
        self.mapper = JointMapper(self.schema)

        self.latent_dim = latent_dim
        self.action_dim = self.schema.action_dim
        self.residual_scale = float(residual_scale)

        self.reflex_dim = self.obs_schema.dim("reflex")
        self.phase_dim = int(phase_dim)
        self.decoder_input_dim = self.reflex_dim + self.phase_dim + self.latent_dim

        self._zero_reflex = np.zeros((self.reflex_dim,), dtype=np.float32)
        self._zero_phase = np.zeros((self.phase_dim,), dtype=np.float32)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        basis_specs = [
            ("leg_l", "X"),
            ("leg_l", "Y"),
            ("leg_l", "Z"),
            ("leg_r", "X"),
            ("leg_r", "Y"),
            ("leg_r", "Z"),
            ("knee_l", "X"),
            ("knee_r", "X"),
            ("ankle_l", "X"),
            ("ankle_r", "X"),
            ("spine", "X"),
            ("spine", "Y"),
            ("spine", "Z"),
            ("tail", "X"),
            ("tail", "Y"),
        ]

        self.basis_projector = BasisProjector(self.schema, basis_specs=basis_specs)

        self.reflex_idx = reflex_idx
        self.reflex_clip = float(reflex_clip)

        if self.reflex_idx is not None:
            if len(self.reflex_idx) == 0:
                raise ValueError("reflex_idx is empty")
            if min(self.reflex_idx) < 0 or max(self.reflex_idx) >= self.reflex_dim:
                raise ValueError(
                    f"reflex_idx out of range: valid [0, {self.reflex_dim-1}]"
                )
            if self.reflex_idx is not None and len(self.reflex_idx) != self.reflex_dim:
                raise ValueError(
                    "For now reflex_idx must select all dims (len == reflex_dim)."
                )

        idx = (
            list(range(self.reflex_dim))
            if self.reflex_idx is None
            else list(self.reflex_idx)
        )
        self.reflex_projector = ReflexProjector(idx=idx, clip=self.reflex_clip)

        self.decoder = StructuredDecoder(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            input_dim=self.decoder_input_dim,
            residual_dim=self.basis_projector.coeff_dim,
            cpg_dim=cpg_dim,
        ).to(self.device)

        if decoder_ckpt is not None:
            sd = torch.load(decoder_ckpt, map_location=self.device)
            self.decoder.load_state_dict(sd, strict=True)

        self.reset()

    def reset(self):
        self.cpg.reset()

    def build_decoder_input(
        self,
        cmd,
        reflex: np.ndarray | None = None,
        phase: np.ndarray | None = None,
    ) -> np.ndarray:
        cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
        if cmd.shape[0] != self.latent_dim:
            raise ValueError(
                f"cmd dim mismatch: got {cmd.shape[0]} expected {self.latent_dim}"
            )

        if reflex is None:
            reflex = self._zero_reflex
        else:
            reflex = np.asarray(reflex, dtype=np.float32).reshape(-1)
            if reflex.shape[0] != self.reflex_dim:
                raise ValueError(
                    f"reflex dim mismatch: got {reflex.shape[0]} expected {self.reflex_dim}"
                )

        if phase is None:
            phase = self._zero_phase
        else:
            phase = np.asarray(phase, dtype=np.float32).reshape(-1)
            if phase.shape[0] != self.phase_dim:
                raise ValueError(
                    f"phase dim mismatch: got {phase.shape[0]} expected {self.phase_dim}"
                )

        x = np.concatenate([reflex, phase, cmd], axis=0).astype(np.float32)
        if x.shape[0] != self.decoder_input_dim:
            raise RuntimeError(
                f"decoder input dim mismatch: got {x.shape[0]} expected {self.decoder_input_dim}"
            )
        return x

    def step(
        self,
        cmd,
        obs=None,
        reflex: np.ndarray | None = None,
        phase: np.ndarray | None = None,
    ):
        if phase is None and self.phase_dim == 4:
            phiL = float(self.cpg.left.phase)
            phiR = float(self.cpg.right.phase)
            phase = np.asarray(
                [np.sin(phiL), np.cos(phiL), np.sin(phiR), np.cos(phiR)],
                dtype=np.float32,
            )

        obs_dict = self._parse_unity_obs(obs)
        reflex_full = self.obs_schema.build_vector(obs_dict, level="reflex")

        reflex = self.reflex_projector(reflex_full)

        x = self.build_decoder_input(cmd, reflex=reflex, phase=phase)

        with torch.no_grad():
            xt = torch.from_numpy(x).to(self.device).unsqueeze(0)
            cpg_params_t, residual_coeff_t = self.decoder.inference(xt)

        cpg_params = cpg_params_t.squeeze(0).cpu().numpy()
        residual_coeff = residual_coeff_t.squeeze(0).cpu().numpy()

        residual = self.basis_projector(residual_coeff)

        cpg_state = self.cpg.step(cpg_params, reflex=reflex)
        base = self.mapper.decode(cpg_state).astype(np.float32)

        out = base + self.residual_scale * residual
        return np.clip(out, -1.0, 1.0)

    def _parse_unity_obs(self, obs_1d: np.ndarray) -> dict:
        x = np.asarray(obs_1d, dtype=np.float32).reshape(-1)
        i = 0

        if x.size < 29:
            raise ValueError(f"Observation too small: {x.size} < 29")

        hipY = x[i]
        i += 1
        vel = x[i : i + 3]
        i += 3
        up = x[i : i + 3]
        i += 3
        fwd = x[i : i + 3]
        i += 3
        com = x[i : i + 3]
        i += 3

        toes = x[i : i + 16]

        return {
            "hipY": np.array([hipY], dtype=np.float32),
            "com": com,
            "toes": toes,
        }
