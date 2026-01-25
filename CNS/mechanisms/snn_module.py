import torch
import torch.nn as nn
import numpy as np


class IzhikevichLayer(nn.Module):
    def __init__(self, input_size, output_size, a=0.02, b=0.2, c=-65.0, d=8.0):
        super().__init__()
        
        self.synapse = nn.Linear(input_size, output_size, bias=False)
        
        self.a = nn.Parameter(torch.ones(output_size) * a + torch.randn(output_size) * 0.002, requires_grad=False)
        self.b = nn.Parameter(torch.ones(output_size) * b + torch.randn(output_size) * 0.02, requires_grad=False)
        self.c = nn.Parameter(torch.ones(output_size) * c + torch.randn(output_size) * 0.01, requires_grad=False)
        self.d = nn.Parameter(torch.ones(output_size) * d + torch.randn(output_size) * 0.01, requires_grad=False)
        
        self.v = None 
        self.u = None 
        self.trace = None 
        self.pre_spike = None
        self.post_spike = None
        self.trace_decay = 0.95

    def reset_state(self, batch_size, device):
        
        self.v = torch.ones(batch_size, self.synapse.out_features).to(device) * -65.0
        self.u = self.v * self.b
        self.trace = torch.zeros_like(self.v)
        self.post_spike = torch.zeros(batch_size, self.synapse.out_features).to(device)

    def forward(self, external_current, dt=1.0):
        
        self.pre_spike = external_current.detach()
        I_rec = self.synapse(self.post_spike)
        
        I = external_current + I_rec

        if self.v is None:
            self.reset_state(external_current.shape[0], external_current.device)
            
        # v' = 0.04v^2 + 5v + 140 - u + I
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        v_next = self.v + dt * dv
        v_next = torch.clamp(v_next, min=-100.0, max=100.0)

        # u' = a(bv - u)
        du = self.a * (self.b * self.v - self.u)
        u_next = self.u + dt * du

        self.v = v_next
        self.u = u_next
        
        self.post_spike = (self.v >= 30.0).float()
        
        # v = (1 - spike) * v + spike * c
        self.v = (1.0 - self.post_spike) * self.v + self.post_spike * self.c
        
        # u = u + spike * d
        self.u = self.u + self.post_spike * self.d
        
        self.trace = self.trace * self.trace_decay + self.post_spike
        
        return self.post_spike