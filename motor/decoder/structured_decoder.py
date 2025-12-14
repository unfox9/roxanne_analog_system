import torch
import torch.nn as nn
from typing import Optional, Tuple

import utils


class StructuredDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        input_dim: Optional[int] = None,
        cpg_dim: Optional[int] = None,
        cmd_slice: Optional[Tuple[int, int]] = None,
        residual_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.input_dim = int(input_dim) if input_dim is not None else int(latent_dim)
        self.residual_dim = (
            int(residual_dim) if residual_dim is not None else self.action_dim
        )
        self.cpg_dim = int(cpg_dim) if cpg_dim is not None else int(latent_dim)

        if self.input_dim < self.latent_dim:
            raise ValueError(
                f"input_dim({self.input_dim}) must be >= latent_dim({self.latent_dim})"
            )

        if cmd_slice is None:
            self.cmd_start = self.input_dim - self.latent_dim
            self.cmd_end = self.input_dim
        else:
            self.cmd_start, self.cmd_end = map(int, cmd_slice)
            if not (0 <= self.cmd_start < self.cmd_end <= self.input_dim):
                raise ValueError(
                    f"bad cmd_slice={cmd_slice} for input_dim={self.input_dim}"
                )
            if (self.cmd_end - self.cmd_start) != self.latent_dim:
                raise ValueError(
                    f"cmd_slice width ({self.cmd_end - self.cmd_start}) "
                    f"must equal latent_dim({self.latent_dim})"
                )

        self.trunk = utils.MLP(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_depth=2
        )

        self.cpg_skip = nn.Linear(self.latent_dim, self.cpg_dim, bias=False)
        self.cpg_head = nn.Linear(hidden_dim, self.cpg_dim)

        self.res_head = nn.Linear(hidden_dim, self.residual_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.cpg_skip.weight)
        with torch.no_grad():
            for i in range(min(self.cpg_dim, self.latent_dim)):
                self.cpg_skip.weight[i, i] = 1.0

        nn.init.zeros_(self.cpg_head.weight)
        nn.init.zeros_(self.cpg_head.bias)

        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)

    def _extract_cmd(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.cmd_start : self.cmd_end]

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"x last dim {x.shape[-1]} != input_dim {self.input_dim}")

        cmd = self._extract_cmd(x)

        h = self.trunk(x)
        cpg = torch.tanh(self.cpg_head(h) + self.cpg_skip(cmd))
        res = torch.tanh(self.res_head(h))
        return cpg, res

    @torch.no_grad()
    def inference(self, x: torch.Tensor):
        return self.forward(x)
