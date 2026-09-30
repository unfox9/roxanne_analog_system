from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn


class Synapse(nn.Module):
    """Conductance-based synapse with caller-defined connectivity.

    ``Synapse`` is responsible for synaptic dynamics only: delay,
    conductance decay, reversal potentials, weights, and Dale's principle.
    Network topology can be supplied explicitly through ``connection_mask``.

    Mask convention
    ---------------
    ``connection_mask`` has shape ``[n_post, n_pre]``. A non-zero entry at
    ``[post, pre]`` means that the presynaptic neuron is connected to the
    postsynaptic neuron. The sign of the synaptic effect is *not* encoded in
    the mask. It is determined by the presynaptic neuron's ``neuron_type``:

    - ``+1`` presynaptic neuron -> excitatory conductance / ``e_exc``
    - ``-1`` presynaptic neuron -> inhibitory conductance / ``e_inh``

    If no explicit mask is supplied, a generic random mask is created using
    ``density`` with no E/I topology restriction.

    The legacy ``is_recurrent``, ``is_feedback`` and
    ``allow_inhibitory_feedforward`` arguments are kept temporarily so older
    project code still loads, but new circuits should pass ``connection_mask``
    explicitly and keep topology policy outside this class.
    """

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
        g_scale=1.0,
        enable_plasticity=True,
        pruning_threshold=0.001,
        sprouting_prob=0.01,
        connection_mask=None,
        allowed_mask=None,
        # Deprecated topology-policy arguments. Keep for compatibility only.
        is_recurrent=False,
        is_feedback=False,
        allow_inhibitory_feedforward=False,
    ):
        super().__init__()

        if not 0.0 <= float(density) <= 1.0:
            raise ValueError("density must be in [0, 1]")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if tau_g <= 0.0:
            raise ValueError("tau_g must be > 0")
        if max_delay_ms < dt:
            raise ValueError("max_delay_ms must be >= dt")

        _layer = layer
        _pre = pre_layer if pre_layer is not None else _layer
        _post = post_layer if post_layer is not None else _layer
        if _pre is None or _post is None:
            raise ValueError("Provide layer=... or both pre_layer=... and post_layer=...")

        self.n_pre = int(_pre.n_neurons)
        self.n_post = int(_post.n_neurons)

        self.register_buffer("neuron_type", _pre.neuron_type.clone())
        self.register_buffer("post_neuron_type", _post.neuron_type.clone())

        legacy_requested = bool(
            is_recurrent or is_feedback or allow_inhibitory_feedforward
        )
        if connection_mask is not None and legacy_requested:
            raise ValueError(
                "connection_mask already defines topology; do not combine it "
                "with is_recurrent/is_feedback/allow_inhibitory_feedforward"
            )

        mask, legacy_allowed = self._build_connection_mask(
            density=float(density),
            connection_mask=connection_mask,
            is_recurrent=bool(is_recurrent),
            is_feedback=bool(is_feedback),
            allow_inhibitory_feedforward=bool(allow_inhibitory_feedforward),
        )

        if allowed_mask is not None:
            allowed = self._normalize_mask(
                allowed_mask, self.n_post, self.n_pre, name="allowed_mask"
            )
            if torch.any((mask > 0) & (allowed <= 0)):
                raise ValueError(
                    "connection_mask contains connections forbidden by allowed_mask"
                )
        elif connection_mask is not None:
            # An explicit connection map means the caller owns the topology.
            # Structural plasticity may restore pruned edges, but it will not
            # invent new anatomical pathways outside that map unless a broader
            # allowed_mask is supplied explicitly.
            allowed = mask.clone()
        else:
            allowed = legacy_allowed

        self.register_buffer("mask", mask)
        self.register_buffer("allowed_mask", allowed)

        self.max_weight = float(max_weight)
        self.weight = nn.Parameter(
            torch.empty(self.n_post, self.n_pre, device=self.neuron_type.device)
        )

        # Scale initial weights by the actual average fan-in rather than by a
        # topology assumption hidden inside Synapse.
        mean_fan_in = float(mask.sum(dim=1).float().mean().item())
        effective_n = max(1.0, mean_fan_in)
        w_scale = float(w_init_multiplier) / np.sqrt(effective_n)
        nn.init.uniform_(self.weight, a=0.0, b=w_scale * 5.0)

        # Magnitude initialization is topology-independent. Dale's principle
        # gives inhibitory presynaptic columns a negative stored weight.
        with torch.no_grad():
            inh_cols = self.neuron_type < 0
            if inh_cols.any():
                n_inh_cols = int(inh_cols.sum().item())
                small_inh_weights = torch.empty(
                    self.n_post,
                    n_inh_cols,
                    device=self.weight.device,
                ).uniform_(0.0, w_scale * 0.5)
                self.weight.data[:, inh_cols] = -small_inh_weights

        # Keep at least two buffer slots so randint(1, high) is always valid.
        # ``+ 1`` makes max_delay_ms inclusive when it is an exact multiple of dt.
        self.max_delay_steps = max(2, int(max_delay_ms / dt) + 1)
        self.register_buffer(
            "delays", torch.randint(
                1, self.max_delay_steps, (self.n_pre,), device=self.neuron_type.device
            )
        )

        # Shape: (Batch, Neurons, Time_Window)
        self.spike_buffer = None
        self.buffer_ptr = 0

        self.tau_g = float(tau_g)
        self.decay = np.exp(-dt / tau_g)

        self.e_exc = float(e_exc)
        self.e_inh = float(e_inh)

        self.g_scale = float(g_scale)
        self.g_exc = None
        self.g_inh = None

        self.enable_plasticity = bool(enable_plasticity)
        self.sprouting_prob = float(sprouting_prob)
        self.pruning_threshold = float(pruning_threshold)

        # If the caller gives an exact mask and no broader allowed_mask, every
        # allowed edge is anatomical, so structural plasticity targets 100% of
        # those allowed edges. Otherwise density retains its original meaning.
        self.target_density = (
            1.0 if connection_mask is not None and allowed_mask is None else float(density)
        )

        self.I = None

        self.enforce_dale_principle()

    def _normalize_mask(self, mask, n_post: int, n_pre: int, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(mask, dtype=torch.float32)
        expected = (n_post, n_pre)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return (tensor != 0).to(
            device=self.neuron_type.device, dtype=torch.float32
        )

    def _build_connection_mask(
        self,
        *,
        density: float,
        connection_mask,
        is_recurrent: bool,
        is_feedback: bool,
        allow_inhibitory_feedforward: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if connection_mask is not None:
            mask = self._normalize_mask(
                connection_mask,
                self.n_post,
                self.n_pre,
                name="connection_mask",
            )
            return mask, mask.clone()

        random_matrix = torch.rand(
            self.n_post, self.n_pre, device=self.neuron_type.device
        )

        # Compatibility path for old project code. New code should define the
        # topology before constructing Synapse and pass connection_mask.
        if is_recurrent or is_feedback or allow_inhibitory_feedforward:
            warnings.warn(
                "Synapse topology flags are deprecated. Build a connection_mask "
                "in the circuit/network initialization instead.",
                DeprecationWarning,
                stacklevel=3,
            )

            mask = torch.zeros(
                self.n_post, self.n_pre, dtype=torch.float32,
                device=self.neuron_type.device,
            )
            allowed = torch.zeros_like(mask)
            pre_is_exc = self.neuron_type > 0
            pre_is_inh = self.neuron_type < 0
            post_is_inh = self.post_neuron_type < 0

            if is_recurrent:
                allowed[:, pre_is_exc] = 1.0
                allowed[:, pre_is_inh] = 1.0
                mask[:, pre_is_exc] = (
                    random_matrix[:, pre_is_exc] < density
                ).float()
                mask[:, pre_is_inh] = 1.0
            elif is_feedback:
                valid_feedback = post_is_inh.unsqueeze(1) & pre_is_exc.unsqueeze(0)
                allowed[valid_feedback] = 1.0
                mask[valid_feedback] = (
                    random_matrix[valid_feedback] < density
                ).float()
            else:
                allowed[:, pre_is_exc] = 1.0
                mask[:, pre_is_exc] = (
                    random_matrix[:, pre_is_exc] < density
                ).float()
                if allow_inhibitory_feedforward:
                    allowed[:, pre_is_inh] = 1.0
                    mask[:, pre_is_inh] = (
                        random_matrix[:, pre_is_inh] < density
                    ).float()
            return mask, allowed

        # New default: Synapse itself has no anatomical E/I policy.
        allowed = torch.ones(
            self.n_post, self.n_pre, dtype=torch.float32,
            device=self.neuron_type.device,
        )
        mask = (random_matrix < density).to(torch.float32)
        return mask, allowed

    def reset_state(self, batch_size, device):
        self.g_exc = torch.zeros(batch_size, self.n_post, device=device)
        self.g_inh = torch.zeros(batch_size, self.n_post, device=device)
        self.spike_buffer = torch.zeros(
            batch_size, self.n_pre, self.max_delay_steps, device=device
        )
        self.buffer_ptr = 0

    def evolve_connectivity(self):
        """Prune/sprout only inside the caller-approved ``allowed_mask``."""
        if not self.enable_plasticity:
            return

        with torch.no_grad():
            active = self.mask > 0
            allowed = self.allowed_mask > 0
            weak_synapses = active & (self.weight.data.abs() < self.pruning_threshold)
            self.mask[weak_synapses] = 0.0
            self.weight.data[weak_synapses] = 0.0

            n_allowed = int(allowed.sum().item())
            if n_allowed == 0:
                return

            current = (self.mask > 0) & allowed
            n_current = int(current.sum().item())
            target = int(round(self.target_density * n_allowed))
            n_to_sprout = max(0, target - n_current)

            if n_to_sprout > 0:
                candidates = allowed & (self.mask == 0)
                if self.sprouting_prob < 1.0:
                    candidates = candidates & (
                        torch.rand_like(self.mask) < self.sprouting_prob
                    )
                empty_indices = torch.nonzero(candidates, as_tuple=False)
                if empty_indices.numel() > 0:
                    if empty_indices.size(0) > n_to_sprout:
                        idx = torch.randperm(empty_indices.size(0))[:n_to_sprout]
                        empty_indices = empty_indices[idx]

                    rows = empty_indices[:, 0]
                    cols = empty_indices[:, 1]
                    base_val = 0.01 * self.max_weight
                    noise = (
                        torch.rand(len(rows), device=self.weight.device)
                        * 0.005
                    )
                    magnitude = base_val + noise
                    sign = torch.where(
                        self.neuron_type[cols] > 0,
                        torch.ones_like(magnitude),
                        -torch.ones_like(magnitude),
                    )
                    self.weight.data[rows, cols] = magnitude * sign
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
        self.g_exc = torch.clamp(
            self.g_exc * self.decay + g_exc_influx, min=0, max=10.0
        )
        self.g_inh = torch.clamp(
            self.g_inh * self.decay + g_inh_influx, min=0, max=10.0
        )

        I_exc = self.g_exc * self.g_scale * (self.e_exc - v_post)
        I_inh = self.g_inh * self.g_scale * (self.e_inh - v_post)

        self.I = torch.clamp(I_exc + I_inh, min=-30.0, max=50.0)

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
