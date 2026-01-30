import torch
import torch.nn as nn
import numpy as np


class Synapse(nn.Module):
    def __init__(self,
                 layer=None,
                 pre_layer=None,
                 post_layer=None,
                 density=0.1,
                 dt=1.0, 
                 tau_g=0.5, 
                 max_delay_ms=20,
                 max_weight=1.0,
                 enable_plasticity=True,
                 pruning_threshold=0.01,
                 sprouting_prob=0.01
                 ):
        super().__init__()
        self.layer = layer
        self.n_neurons = layer.n_neurons
        self.pre_layer =  pre_layer if pre_layer is not None else self.layer
        self.post_layer = post_layer if post_layer is not None else self.layer

        self.n_pre = self.pre_layer.n_neurons
        self.n_post = self.post_layer.n_neurons

        self.max_weight = max_weight

        self.register_buffer('neuron_type', self.pre_layer.neuron_type.clone())

        self.weight = nn.Parameter(torch.empty(self.n_post, self.n_pre))
        nn.init.xavier_uniform_(self.weight)

        mask = (torch.rand(self.n_post, self.n_pre) < density).float()
        self.register_buffer('mask', mask)

        self.max_delay_steps = int(max_delay_ms / dt)
        self.register_buffer('delays', torch.randint(1, self.max_delay_steps, (self.n_pre,)))

        # Shape: (Batch, Neurons, Time_Window)
        self.spike_buffer = None
        self.buffer_ptr = 0

        self.tau_g = tau_g
        self.decay = np.exp(-dt / tau_g)

        self.g = None

        self.eneable_plasticity = enable_plasticity
        self.pruning_prob = sprouting_prob
        self.pruning_threshold = pruning_threshold

        self.I = None

        self.enforce_dale_principle()

    def reset_state(self, batch_size, device):
        self.spike_buffer = torch.zeros(
            batch_size, self.n_pre, self.max_delay_steps, device=device
        )
        self.buffer_ptr = 0

    def evolve_connectivity(self):
        if not self.enable_plasticity:
            return
        with torch.no_grad():
            weak_synapses = (self.mask == 1) & (self.weight.data.abs() < self.pruning_threshold)

            self.mask[weak_synapses] = 0.0
            self.weight.data[weak_synapses] = 0.0

            current_density = self.mask.sum() / self.mask.numel()

            if current_density < self.target_density:
                n_total = self.mask.numel()
                n_target = int(self.target_density * n_total)
                n_current = int(self.mask.sum().item())
                n_to_sprout = max(0, n_target - n_current)

                if n_to_sprout > 0:
                    empty_indices = torch.nonzero(self.mask == 0, as_tuple=False)

                    if empty_indices.size(0) > n_to_sprout:
                        idx = torch.randperm(empty_indices(0))[:n_to_sprout]
                        new_conns = empty_indices[idx]

                        rows = new_conns[:, 0]
                        cols = new_conns[:, 1]

                        current_types = self.pre_neuron_type[cols]

                        base_val = 0.01 * self.max_weight
                        noise = torch.rand(n_to_sprout, device=self.weight.device) * 0.005

                        abs_weights = base_val + noise

                        signs = torch.where(current_types > 0, 1.0, -1.0)

                        self.weight.data[rows, cols] = abs_weights * signs

                        self.mask[rows, cols] = 1.0
            
            self.enforce_dale_principle()

    def forward(self, pre_spikes, v_post):
        batch_size = pre_spikes.shape[0]

        if self.spike_buffer is None:
            self.reset_state(batch_size, pre_spikes.device)

        self.spike_buffer[:, :, self.buffer_ptr] = pre_spikes

        read_indices = (self.buffer_ptr - self.delays) % self.max_delay_steps
        idx_tensor = read_indices.view(1, -1, 1).expand(batch_size, -1, 1)
        delayed_spikes = torch.gather(self.spike_buffer, 2, idx_tensor).squeeze(2)

        self.buffer_ptr = (self.buffer_ptr + 1) % self.max_delay_steps

        effective_weight = self.weight * self.mask

        g_influx = torch.matmul(delayed_spikes, effective_weight.T)

        self.g = self.g * self.decay + g_influx

        I_total = torch.zeros_like(v_post)

        exc_mask = (self.g > 0)
        if exc_mask.any():
            I_total[exc_mask] = self.g[exc_mask] * (0.0 - v_post[exc_mask])

        inh_mask = (self.g < 0)
        if inh_mask.any():
            I_total[inh_mask] = (-self.g[inh_mask]) * (-80.0 - v_post[inh_mask])

        self.I = I_total
        return self.I
    
    def enforce_dale_principle(self):
        with torch.no_grad():
            self.weight.data *= self.mask
            exc_mask = (self.neuron_type > 0)
            if exc_mask.any():
                self.weight.data[:, exc_mask] = self.weight.data[
                    :, exc_mask
                ].clamp(min=0.0, max=self.max_weight)

            inh_mask = (self.neuron_type < 0)
            if inh_mask.any():
                self.weight.data[:, inh_mask] = self.weight.data[
                    :, inh_mask
                ].clamp(min=-self.max_weight, max=0.0)
