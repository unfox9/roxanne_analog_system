import torch
import torch.nn as nn
import numpy as np
from CNS.mechanisms.snn_synapse import Synapse


class IzhikevichLayer(nn.Module):
    def __init__(
        self,
        n_neurons,
        synapse=None,
        exc_ratio=0.8,
        a_exc=0.02,
        b_exc=0.2,
        c_exc=-65.0,
        d_exc=8.0,
        a_inh=0.1,
        b_inh=0.2,
        c_inh=-65.0,
        d_inh=2.0,
        dt=1.0,
        refractory_steps=5,
        enable_alic=True,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.dt = dt
        self.refractory_steps = refractory_steps

        self.v_threshold = 30.0

        # Dale's Principle
        # Excitatory and Inhibitory Neurons
        self.n_exc = int(n_neurons * exc_ratio)
        self.n_inh = n_neurons - self.n_exc

        neuron_type = torch.ones(n_neurons)
        neuron_type[self.n_exc :] = -1.0
        self.register_buffer("neuron_type", neuron_type)

        self.a_exc = a_exc
        self.b_exc = b_exc
        self.c_exc = c_exc
        self.d_exc = d_exc

        self.a_inh = a_inh
        self.b_inh = b_inh
        self.c_inh = c_inh
        self.d_inh = d_inh

        _a = torch.zeros(n_neurons)
        _b = torch.zeros(n_neurons)
        _c = torch.zeros(n_neurons)
        _d = torch.zeros(n_neurons)

        _a[: self.n_exc] = self.a_exc + torch.randn(self.n_exc) * 0.002
        _b[: self.n_exc] = self.b_exc + torch.randn(self.n_exc) * 0.002
        _c[: self.n_exc] = self.c_exc + torch.randn(self.n_exc) * 1.0
        _d[: self.n_exc] = self.d_exc + torch.randn(self.n_exc) * 1.0

        _a[self.n_exc :] = self.a_inh + torch.randn(self.n_inh) * 0.01
        _b[self.n_exc :] = self.b_inh + torch.randn(self.n_inh) * 0.002
        _c[self.n_exc :] = self.c_inh + torch.randn(self.n_inh) * 1.0
        _d[self.n_exc :] = self.d_inh + torch.randn(self.n_inh) * 0.5

        self.a = nn.Parameter(_a, requires_grad=False)
        self.b = nn.Parameter(_b, requires_grad=False)
        self.c = nn.Parameter(_c, requires_grad=False)
        self.d = nn.Parameter(_d, requires_grad=False)

        self.v = None
        self.u = None

        # Post-synaptic Calcium
        self.calcium = None
        self.tau_ca = 20.0
        self.ca_pre = 0.0
        self.ca_spike = 1.0

        self.pre_spike = None
        self.post_spike = None

        self.refractory_counter = None

        self.neuro_sensitivity = nn.Parameter(
            torch.rand(n_neurons) * 1.5 + 0.1, requires_grad=False
        )
        self.noise_level = nn.Parameter(
            torch.rand(n_neurons) * 2.0 + 0.2, requires_grad=False
        )

        self.bias_current = nn.Parameter(
            torch.ones(n_neurons) * 5.0, requires_grad=False
        )
        self.target_rate = 0.1
        self.bias_lr = 0.002

        self.synapse = synapse

        self.enable_alic = enable_alic

        self.I = None

    def reset_state(self, batch_size, device):
        self.v = torch.ones(batch_size, self.n_neurons).to(device) * -65.0
        self.u = self.v * self.b
        self.I = torch.zeros(batch_size, self.n_neurons).to(device)
        self.calcium = torch.zeros(batch_size, self.n_neurons).to(device)
        self.refractory_counter = torch.zeros(batch_size, self.n_neurons).to(device)
        self.post_spike = torch.zeros(batch_size, self.n_neurons).to(device)
        self.pre_spike = torch.zeros(batch_size, self.n_neurons).to(device)

    def forward(self, total_input_current, neuromodulation_deltas=None):
        if self.v is None:
            self.reset_state(total_input_current.shape[0], total_input_current.device)
        prev_spike = (
            self.pre_spike
            if self.pre_spike is not None
            else torch.zeros_like(self.post_spike)
        )

        bias_effect = 0.0
        current_d = self.d
        current_b = self.b
        if neuromodulation_deltas:
            if "input_bias" in neuromodulation_deltas:
                global_bias = neuromodulation_deltas["input_bias"]
                bias_effect = global_bias * self.neuro_sensitivity.to(
                    total_input_current.device
                )
            if "d" in neuromodulation_deltas:
                current_d = self.d + neuromodulation_deltas["d"]
            if "b" in neuromodulation_deltas:
                current_b = self.b + neuromodulation_deltas["b"]

        background_noise = torch.randn_like(total_input_current) * self.noise_level.to(
            total_input_current.device
        )

        self.I = (
            total_input_current + bias_effect + background_noise + self.bias_current
        )
        if self.synapse:
            recurrent_current = self.synapse(prev_spike, self.v)
            self.I = self.I + recurrent_current

        # ALIC - Activity-Level Informed Competition
        if hasattr(self, 'enable_alic') and self.enable_alic:
            can_compete = (self.refractory_counter <= 0).float()
            competing_I = self.I * can_compete + (-9999.0) * (1.0 - can_compete)
            i_max = competing_I.max(dim=1, keepdim=True)[0]
            i_thresh = i_max / 2.0
            if i_max.max().item() > 0:
                i_thresh = i_max / 2.0
                is_follower = (self.I < i_max - 1e-4).float() * can_compete
                competing_mask  = (self.I > i_thresh).float() * is_follower
                alpha_inh = 1.625
                dynamic_inhibition = alpha_inh * i_max * competing_mask
                self.I = self.I - dynamic_inhibition

        self.I = torch.clamp(self.I, -30, 50)

        d_calcium = (
            -self.calcium / self.tau_ca
        ) * self.dt + self.post_spike * self.ca_spike
        self.calcium = self.calcium + d_calcium

        is_refractory = (self.refractory_counter > 0).float()
        # v' = 0.04v^2 + 5v + 140 - u + I
        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + self.I
        dv = dv * (1.0 - is_refractory)  
        v_next = self.v + self.dt * dv
        v_next = torch.clamp(v_next, min=-100.0, max=100.0)

        # u' = a(bv - u)
        du = self.a * (current_b * self.v - self.u)
        du = du * (1.0 - is_refractory)
        u_next = self.u + self.dt * du

        self.v = v_next
        self.u = u_next

        can_fire = (self.refractory_counter <= 0).float()
        spike_condition = (self.v >= self.v_threshold).float()
        self.post_spike = spike_condition * can_fire

        with torch.no_grad():
            activity_error = (self.post_spike - self.target_rate).mean(dim=0)
            self.bias_current -= self.bias_lr * activity_error
            self.bias_current.clamp_(min=-5.0, max=20.0)

        # v = (1 - spike) * v + spike * c
        self.v = (1.0 - self.post_spike) * self.v + self.post_spike * self.c
        # u = u + spike * d
        self.u = self.u + self.post_spike * current_d

        # Update refractory counter
        self.refractory_counter = torch.where(
            self.post_spike > 0,
            torch.full_like(self.refractory_counter, self.refractory_steps),
            self.refractory_counter - 1,
        )
        self.refractory_counter = torch.clamp(self.refractory_counter, min=0)

        self.pre_spike = self.post_spike.clone()

        self.v = self.v.detach()
        self.u = self.u.detach()
        self.calcium = self.calcium.detach()
        self.refractory_counter = self.refractory_counter.detach()
        self.post_spike = self.post_spike.detach()
        self.pre_spike = self.pre_spike.detach()

        return self.post_spike, self.calcium
