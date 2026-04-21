import torch
import torch.nn as nn
import numpy as np


class Synapse(nn.Module):
    def __init__(
        self,
        layer=None,
        pre_layer=None,
        post_layer=None,
        density=0.3,
        dt=1.0,
        tau_g=5.0,
        e_exc=0.0,
        e_inh=-80.0,
        max_delay_ms=20,
        max_weight=1.0,
        w_init_multiplier=3.0,
        g_scale=0.01,
        enable_plasticity=True,
        pruning_threshold=0.001,
        sprouting_prob=0.01,
        is_recurrent=False,
        is_feedback=False
    ):
        super().__init__()
        _layer = layer
        _pre = pre_layer if pre_layer is not None else _layer
        _post = post_layer if post_layer is not None else _layer

        self.n_pre = _pre.n_neurons
        self.n_post = _post.n_neurons

        self.register_buffer("neuron_type", _pre.neuron_type.clone())
        self.register_buffer("post_neuron_type", _post.neuron_type.clone())

        self.max_weight = max_weight
        self.weight = nn.Parameter(torch.empty(self.n_post, self.n_pre))
        effective_n = max(1.0, self.n_pre * density)
        w_scale = w_init_multiplier / np.sqrt(effective_n)
        nn.init.uniform_(self.weight, a=0.0, b=w_scale * 5.0)
        with torch.no_grad():
            _temp_inh_mask = self.neuron_type < 0
            if _temp_inh_mask.any():
                n_inh_cols = _temp_inh_mask.sum().item()
                small_inh_weights = torch.empty(self.n_post, n_inh_cols, device=self.weight.device).uniform_(0.0, w_scale * 0.5)
                self.weight.data[:, _temp_inh_mask] = small_inh_weights * -1.0
        mask = torch.zeros_like(self.weight)
        pre_is_exc = self.neuron_type > 0
        pre_is_inh = self.neuron_type < 0
        post_is_exc = self.post_neuron_type > 0
        post_is_inh = self.post_neuron_type < 0
        random_matrix = torch.rand(self.n_post, self.n_pre)
        if is_recurrent:
            mask[:, pre_is_exc] = (random_matrix[:, pre_is_exc] < density).float()
            mask[:, pre_is_inh] = 1.0
        elif is_feedback:
            valid_feedback = post_is_inh.unsqueeze(1) & pre_is_exc.unsqueeze(0)
            mask[valid_feedback] = (random_matrix[valid_feedback] < density).float()
        else:
            mask[:, pre_is_exc] = (random_matrix[:, pre_is_exc] < density).float()
            mask[:, pre_is_inh] = 0.0
        self.register_buffer("mask", mask)
   
        
        self.max_delay_steps = int(max_delay_ms / dt)
        self.register_buffer(
            "delays", torch.randint(1, self.max_delay_steps, (self.n_pre,))
        )

        # Shape: (Batch, Neurons, Time_Window)
        self.spike_buffer = None
        self.buffer_ptr = 0

        self.tau_g = tau_g
        self.decay = np.exp(-dt / tau_g)

        self.e_exc = e_exc
        self.e_inh = e_inh

        self.g_scale = g_scale
        self.g_exc = None
        self.g_inh = None

        self.enable_plasticity = enable_plasticity
        self.sprouting_prob = sprouting_prob
        self.pruning_threshold = pruning_threshold
        self.target_density = density

        self.I = None

        self.enforce_dale_principle()

    def reset_state(self, batch_size, device):
        self.g_exc = torch.zeros(batch_size, self.n_post, device=device)
        self.g_inh = torch.zeros(batch_size, self.n_post, device=device)
        self.spike_buffer = torch.zeros(
            batch_size, self.n_pre, self.max_delay_steps, device=device
        )
        self.buffer_ptr = 0

    def evolve_connectivity(self):
        if not self.enable_plasticity:
            return
        with torch.no_grad():
            exc_pre_mask = (self.neuron_type > 0).unsqueeze(0)
            weak_synapses = (self.mask == 1) & (
                self.weight.data.abs() < self.pruning_threshold
                & exc_pre_mask
            )
            self.mask[weak_synapses] = 0.0
            self.weight.data[weak_synapses] = 0.0
            n_exc_pre = (self.neuron_type > 0).sum().item()
            if n_exc_pre == 0:
                return
            n_total_exc_conns = self.n_post * n_exc_pre
            n_current_exc_conns = (self.mask * exc_pre_mask).sum().item()
            current_density = n_current_exc_conns / n_total_exc_conns
            if current_density < self.target_density:
                n_target = int(self.target_density * n_total_exc_conns)
                n_to_sprout = max(0, n_target - int(n_current_exc_conns))
                if n_to_sprout > 0:
                    empty_mask = (self.mask == 0) & exc_pre_mask
                    empty_indices = torch.nonzero(empty_mask, as_tuple=False)
                    if empty_indices.size(0) > n_to_sprout:
                        idx = torch.randperm(empty_indices.size(0))[:n_to_sprout]
                        new_conns = empty_indices[idx]
                        rows = new_conns[:, 0]
                        cols = new_conns[:, 1]
                        base_val = 0.01 * self.max_weight
                        noise = (
                            torch.rand(n_to_sprout, device=self.weight.device) * 0.005
                        )
                        abs_weights = base_val + noise
                        self.weight.data[rows, cols] = abs_weights
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

        w_abs = self.weight.abs() * self.mask

        exc_mask = (self.neuron_type > 0).unsqueeze(0)
        inh_mask = (self.neuron_type < 0).unsqueeze(0)

        w_exc_magnitude = w_abs * exc_mask
        w_inh_magnitude = w_abs * inh_mask

        g_exc_influx = torch.matmul(delayed_spikes, w_exc_magnitude.T)
        g_inh_influx = torch.matmul(delayed_spikes, w_inh_magnitude.T)
        self.g_exc = torch.clamp(self.g_exc * self.decay + g_exc_influx, min=0, max=10.0)
        self.g_inh = torch.clamp(self.g_inh * self.decay + g_inh_influx, min=0, max=10.0)

        I_exc = self.g_exc * self.g_scale * (self.e_exc - v_post)
        I_inh = self.g_inh * self.g_scale * (self.e_inh - v_post)

        self.I = I_exc + I_inh
        self.I = torch.clamp(self.I, min=-30.0, max=50.0)

        self.I = self.I.detach()
        self.g_exc = self.g_exc.detach()
        self.g_inh = self.g_inh.detach()
        self.spike_buffer = self.spike_buffer.detach()
        return self.I

    def enforce_dale_principle(self):
        with torch.no_grad():
            self.weight.data *= self.mask
            exc_mask = self.neuron_type > 0
            if exc_mask.any():
                self.weight.data[:, exc_mask] = self.weight.data[:, exc_mask].clamp(
                    min=0.0, max=self.max_weight
                )

            inh_mask = self.neuron_type < 0
            if inh_mask.any():
                self.weight.data[:, inh_mask] = self.weight.data[:, inh_mask].clamp(
                    min=-self.max_weight, max=0.0
                )
