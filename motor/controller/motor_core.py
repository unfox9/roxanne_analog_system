import numpy as np
import torch

from motor.cpg.biped_cpg import BipedCPG
from motor.decoder.joint_mapper import JointMapper
from motor.schema.action_schema import ActionSchema
from motor.decoder.structured_decoder import StructuredDecoder


class MotorController:
    def __init__(
        self,
        mapping_path: str,
        dt: float = 0.02,
        latent_dim: int = 8,
        decoder_ckpt: str | None = None,
        residual_scale: float = 0.10,
        device: str = "cpu",
        expected_action_dim: int | None = None,
    ):
        self.schema = ActionSchema.from_mapping_file(
            mapping_path,
            expected_dim=expected_action_dim,
            strict_contiguous=False,
        )

        self.dt = dt
        self.cpg = BipedCPG(dt=dt)
        self.mapper = JointMapper(self.schema)

        self.latent_dim = latent_dim
        self.action_dim = self.schema.action_dim
        self.residual_scale = float(residual_scale)

        self.device = torch.device(device)
        self.decoder = StructuredDecoder(latent_dim=self.latent_dim, action_dim=self.action_dim).to(self.device)
        self.decoder.eval()

        if decoder_ckpt is not None:
            sd = torch.load(decoder_ckpt, map_location=self.device)
            self.decoder.load_state_dict(sd, strict=True)

        self.reset()

    def reset(self):
        self.cpg.reset()

    def step(self, latent):
        latent = np.asarray(latent, dtype=np.float32).reshape(1, -1)
        assert latent.shape[1] == self.latent_dim

        with torch.no_grad():
            z = torch.from_numpy(latent).to(self.device)
            cpg_params_t, residual_t = self.decoder.inference(z)

        cpg_params = cpg_params_t.squeeze(0).cpu().numpy().astype(np.float32)
        residual = residual_t.squeeze(0).cpu().numpy().astype(np.float32)

        cpg_state = self.cpg.step(cpg_params)
        base = self.mapper.decode(cpg_state).astype(np.float32)

        out = base + self.residual_scale * residual
        return np.clip(out, -1.0, 1.0)
