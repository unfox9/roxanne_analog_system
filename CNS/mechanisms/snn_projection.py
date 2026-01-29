import torch
import torch.nn as nn
import numpy as np

class ConductanceSynapse(nn.Module):
    def __init__(self, n_pre, n_post, dt=1.0, tau_g=0.5, E_exc=0.0, E_inh=-80.0):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(n_post, n_pre))
        nn.init.uniform_(self.weight, a=0.001, b=0.02) 
        
        self.E_exc = E_exc
        self.E_inh = E_inh
        
        self.decay = np.exp(-dt / tau_g)
        
        self.register_buffer("g_exc", torch.zeros(1, n_post)) # Batch size 會自動廣播，初始設 1 沒關係
        self.register_buffer("g_inh", torch.zeros(1, n_post))

    def reset_state(self, batch_size, device):
        self.g_exc = torch.zeros(batch_size, self.weight.shape[0], device=device)
        self.g_inh = torch.zeros(batch_size, self.weight.shape[0], device=device)

    def forward(self, pre_spikes, v_post):
        g_influx = torch.matmul(pre_spikes, self.weight.T)
        
        delta_g_exc = torch.relu(g_influx)
        delta_g_inh = torch.relu(-g_influx)
        
        self.g_exc = self.g_exc * self.decay + delta_g_exc
        self.g_inh = self.g_inh * self.decay + delta_g_inh
        
        I_exc = self.g_exc * (self.E_exc - v_post)
        I_inh = self.g_inh * (self.E_inh - v_post)
        
        return I_exc + I_inh