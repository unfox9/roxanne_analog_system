import torch
import torch.nn as nn


class StructuredDecoder(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.cpg_skip = nn.Linear(latent_dim, latent_dim, bias=False)
        self.cpg_head = nn.Linear(hidden_dim, latent_dim)

        self.res_head = nn.Linear(hidden_dim, action_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.cpg_skip.weight)
        with torch.no_grad():
            for i in range(min(self.latent_dim, self.cpg_skip.weight.shape[0])):
                self.cpg_skip.weight[i, i] = 1.0

        nn.init.zeros_(self.cpg_head.weight)
        nn.init.zeros_(self.cpg_head.bias)

        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)

    @torch.no_grad()
    def inference(self, latent: torch.Tensor):
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        h = self.trunk(latent)
        cpg = torch.tanh(self.cpg_head(h) + self.cpg_skip(latent))
        res = torch.tanh(self.res_head(h))
        return cpg, res
