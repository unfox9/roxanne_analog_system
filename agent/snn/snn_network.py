import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFlayer(nn.Module):
    def __init__(self, input_size, output_size, decay=0.9, threshold=1.0, trace_decay=0.95):
        super().__init__()
        self.synapse = nn.Linear(input_size, output_size, bias=False)

        self.decay = decay
        self.threshold = threshold
        self.trace_decay = trace_decay

        self.v = None 
        self.trace = None

        self.pre_spike = None
        self.post_spike = None

    def reset_state(self):
        self.v = None
        self.trace = None

    def forward(self, x):
        self.pre_spike = x.detach()

        I_t = self.synapse(x)
        
        if self.v is None or self.v.shape != I_t.shape:
            self.v = torch.zeros_like(I_t).to(I_t.device)
            self.trace = torch.zeros_like(I_t).to(I_t.device)

        self.v = self.decay * self.v + I_t

        self.post_spike = (self.v >= self.threshold).float()

        self.v = self.v * (1.0 - self.post_spike)

        self.trace = self.trace_decay * self.trace + self.post_spike

        return self.post_spike
    

class SNN(nn.Module):
    def __init__ (self, input_dim, output_dim, hidden_dims):
        super(SNN, self).__init__()
        self.layer1 = LIFlayer(input_dim, hidden_dims)
        self.layer2 = LIFlayer(hidden_dims, hidden_dims)
        self.layer3 = LIFlayer(hidden_dims, output_dim)

    def reset_states(self):
        self.layer1.reset_state()
        self.layer2.reset_state()
        self.layer3.reset_state()

    def forward(self, x):
        spk1 = self.layer1(x)
        spk2 = self.layer2(spk1)
        spk3 = self.layer3(spk2)
        return spk3
        
