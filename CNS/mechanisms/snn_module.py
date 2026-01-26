import torch
import torch.nn as nn
import numpy as np


class IzhikevichLayer(nn.Module):
    def __init__(self, 
                 n_neurons, 
                 exc_ratio=0.8, 
                 a=0.02, 
                 b=0.2, 
                 c=-65.0, 
                 d=2.0, 
                 max_delay_ms=20, 
                 dt=1.0,
                 refractory_steps=5):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.dt = dt
        self.max_delay_steps = int(max_delay_ms / dt)
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
        
        self.axonal_delays = torch.randint(1, self.max_delay_steps, (n_neurons,))

        self.v = None 
        self.u = None 
        
        # Post-synaptic Calcium
        self.calcium = None
        self.tau_ca = 20.0
        self.ca_pre = 0.0
        self.ca_spike = 1.0

        # Shape: (Batch, Neurons, Time_Window)
        self.spike_buffer = None
        self.post_spike = None

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
        self.calcium = torch.zeros(batch_size, self.n_neurons).to(device)
        self.refractory_counter = torch.zeros(batch_size, self.synapse.out_features).to(device)
        self.axonal_delays = self.axonal_delays.to(device)
        self.post_spike = torch.zeros(batch_size, self.synapse.out_features).to(device)
        self.spike_buffer = torch.zeros(batch_size, self.n_neurons, self.max_delay_steps).to(device)

    def forward(self, external_current):
        if self.v is None:
            self.reset_state(external_current.shape[0], external_current.device)
            
        batch_size = external_current.shape[0]

        # delay_indices shape: (Batch, Neurons, 1)
        delay_indices = self.axonal_delays.view(1, -1, 1).expand(batch_size, -1, 1)
        delayed_spikes = torch.gather(self.spike_buffer, 2, delay_indices).squeeze(2)

        # I_internal = Spikes @ Weights
        internal_current = self.synapse(delayed_spikes)

        I = external_current + internal_current

        last_spike = self.spike_buffer[:, :, 0]

        calcium_influx = last_spike * self.ca_spike
        d_calcium = (-self.calcium / self.tau_ca) * self.dt + calcium_influx
        self.calcium = self.calcium + d_calcium

        # v' = 0.04v^2 + 5v + 140 - u + I
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        v_next = self.v + self.dt * dv
        v_next = torch.clamp(v_next, min=-100.0, max=100.0)

        # u' = a(bv - u)
        du = self.a * (self.b * self.v - self.u)
        u_next = self.u + self.dt * du

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

        # buffer[:, :, 1:] = buffer[:, :, :-1]
        # buffer[:, :, 0] = new_spikes
        self.spike_buffer = torch.roll(self.spike_buffer, shifts=1, dims=2)
        self.spike_buffer[:, :, 0] = self.post_spike

        return self.post_spike, self.calcium

    def enforce_dale_principle(self, max_exc_weight=1.0, max_inh_weight=1.0):
        with torch.no_grad():
            exc_mask = self.neuron_type > 0
            if exc_mask.any():
                self.synapse.weight[:, exc_mask] = self.synapse.weight[:, exc_mask].clamp(
                    min=0.0, max=max_exc_weight
                )
            
            inh_mask = ~exc_mask  # 取反
            if inh_mask.any():
                self.synapse.weight[:, inh_mask] = self.synapse.weight[:, inh_mask].clamp(
                    min=-max_inh_weight, max=0.0
                )