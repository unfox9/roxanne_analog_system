import torch
import torch.nn as nn
import numpy as np


class IzhikevichLayer(nn.Module):
    def __init__(self, n_neurons, exc_ratio=0.8, a=0.02, b=0.2, c=-65.0, d=8.0, refractory_steps=5):
        super().__init__()
        
        self.n_neurons = n_neurons

        self.refractory_steps = refractory_steps
        
        # Dale's Principle
        self.n_exc = int(n_neurons * exc_ratio)  
        self.n_inh = n_neurons - self.n_exc 

        self.neuron_type = torch.ones(n_neurons)
        self.neuron_type[self.n_exc:] = -1.0       

        self.synapse = nn.Linear(n_neurons, n_neurons, bias=False)

        self.a = nn.Parameter(torch.ones(n_neurons) * a + torch.randn(n_neurons) * 0.002, requires_grad=False)
        self.b = nn.Parameter(torch.ones(n_neurons) * b + torch.randn(n_neurons) * 0.02, requires_grad=False)
        self.c = nn.Parameter(torch.ones(n_neurons) * c + torch.randn(n_neurons) * 0.01, requires_grad=False)
        self.d = nn.Parameter(torch.ones(n_neurons) * d + torch.randn(n_neurons) * 0.01, requires_grad=False)
        
        self.v = None 
        self.u = None 
        self.trace = None 
        self.pre_spike = None
        self.post_spike = None
        self.trace_decay = 0.95

        self.refractory_counter = None

    def init_weights(self, strategy="random", connection_prob=0.15, max_weight=1.0):
        with torch.no_grad():
            self.synapse.weight.fill_(0.0)
            if strategy == "random":
                for i in range(self.n_neurons):
                    for j in range(self.n_neurons):
                        if i != j and torch.rand(1) < connection_prob:
                            strength = torch.rand(1) * 0.3 * max_weight
                            self.synapse.weight[j, i] = strength * self.neuron_type[i]

            elif strategy == "empty":
                self.synapse.weight.data.fill_(0.0)

    def reset_state(self, batch_size, device):
        self.v = torch.ones(batch_size, self.synapse.out_features).to(device) * -65.0
        self.u = self.v * self.b
        self.trace = torch.zeros_like(self.v)
        self.post_spike = torch.zeros(batch_size, self.synapse.out_features).to(device)
        self.refractory_counter = torch.zeros(batch_size, self.synapse.out_features).to(device)

    def forward(self, external_current, dt=1.0):
        if self.v is None:
            self.reset_state(external_current.shape[0], external_current.device)
            
        self.pre_spike = external_current.detach()

        # I_internal = Spikes @ Weights
        internal_current = self.synapse(self.post_spike)

        I = external_current + internal_current

        # v' = 0.04v^2 + 5v + 140 - u + I
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        v_next = self.v + dt * dv
        v_next = torch.clamp(v_next, min=-100.0, max=100.0)

        # u' = a(bv - u)
        du = self.a * (self.b * self.v - self.u)
        u_next = self.u + dt * du

        self.v = v_next
        self.u = u_next
        
        can_fire = (self.refractory_counter <= 0).float()
        spike_condition = (self.v >= 30.0).float()
        self.post_spike = spike_condition * can_fire
        
        # v = (1 - spike) * v + spike * c
        self.v = (1.0 - self.post_spike) * self.v + self.post_spike * self.c
        # u = u + spike * d
        self.u = self.u + self.post_spike * self.d

        # Update refractory counter
        self.refractory_counter = torch.where(
            self.post_spike > 0,
            torch.full_like(self.refractory_counter, self.refractory_steps),
            self.refractory_counter - 1
        )
        self.refractory_counter = torch.clamp(self.refractory_counter, min=0)

        self.trace = self.trace * self.trace_decay + self.post_spike
        
        return self.post_spike
    
    def enforce_dale_principle(self, max_exc_weight=1.0, max_inh_weight=1.0):
        with torch.no_grad():
            exc_mask = self.neuron_type > 0
            inh_mask = self.neuron_type < 0
            
            self.synapse.weight[:, exc_mask] = self.synapse.weight[:, exc_mask].clamp(min=0.0, max=max_exc_weight)
            self.synapse.weight[:, inh_mask] = self.synapse.weight[:, inh_mask].clamp(min=-max_inh_weight, max=0.0)