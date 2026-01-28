import torch
import torch.nn as nn


class ThreeFactorOptimizer:
    def __init__(
        self,
        layer,
        n_neurons,
        dopamine,
        lr=1e-3,
        theta_d=0.5,
        theta_p=1.2,
        gamma_p=1.0,
        gamma_d=0.5,
        max_weight=4.0,
    ):
        self.n_neurons = n_neurons
        self.layer = layer
        self.lr = lr
        self.max_weight = max_weight

        self.theta_d = theta_d
        self.theta_p = theta_p

        self.gamma_p = gamma_p
        self.gamma_d = gamma_d

        self.eligibility_trace = torch.zeros(self.n_neurons, self.n_neurons).to(
            layer.synapse.weight.device
        )

        self.pre_trace = torch.zeros(self.n_neurons).to(layer.synapse.weight.device)

        self.v_threshold_for_learning = -50.0

        # STDP parameters
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        self.A_plus = 1.0
        self.A_minus = 0.5

        self.dopamine = dopamine
        self.dopamine_level = 0.0


    def step(self, modulation_deltas=None):
        lr_scale = 1.0
        is_frozen = False
        if modulation_deltas is not None:
            if 'lr_scale' in modulation_deltas:
                lr_scale += modulation_deltas['lr_scale']
            if 'learning_inhibition' in modulation_deltas:
                if modulation_deltas['learning_inhibition'] > 0.5:
                    is_frozen = True
        if is_frozen:
            return

        effective_lr = self.lr * lr_scale

        current_spikes = self.layer.post_spike.detach()
        ca = self.layer.calcium.mean(dim=0)

        if current_spikes is None:
            return

        if self.pre_trace.device != current_spikes.device:
            self.pre_trace = self.pre_trace.to(current_spikes.device)
        if self.eligibility_trace.device != current_spikes.device:
            self.eligibility_trace = self.eligibility_trace.to(current_spikes.device)

        ltp_mask = (ca > self.theta_p).float()
        ltd_mask = ((ca > self.theta_d) & (ca <= self.theta_p)).float()

        pre_activity = self.layer.spike_buffer.mean(dim=2).mean(dim=0)

        # Pre Rate: (1, N)
        pre_rate_row = pre_activity.unsqueeze(0)

        ltp_mask_col = ltp_mask.unsqueeze(1)
        ltd_mask_col = ltd_mask.unsqueeze(1)

        delta_ltp = torch.matmul(ltp_mask_col, pre_rate_row) * self.gamma_p
        delta_ltd = torch.matmul(ltd_mask_col, pre_rate_row) * self.gamma_d

        stdp_update = delta_ltp - delta_ltd

        # E(t) = E(t-1) * decay + STDP(t)
        self.eligibility_trace = self.eligibility_trace * 0.95 + stdp_update

        self.dopamine_level = self.dopamine.signal()

        # dW = Learning_Rate * Dopamine * Eligibility
        if abs(self.dopamine_level) > 0.001:
            delta_w = effective_lr * self.dopamine_level * self.eligibility_trace

            with torch.no_grad():
                self.layer.synapse.weight += delta_w
                self.layer.enforce_dale_principle(
                    max_exc_weight=self.max_weight, max_inh_weight=self.max_weight
                )

        with torch.no_grad():
            self.layer.synapse.weight.mul_(0.9999)


    def get_weight_stats(self):
        w = self.layer.synapse.weight.data
        return {
            "mean": w.mean().item(),
            "std": w.std().item(),
            "max": w.max().item(),
            "min": w.min().item(),
            "nonzero": (w != 0).sum().item(),
        }