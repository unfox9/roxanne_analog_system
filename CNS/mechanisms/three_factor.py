import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ThreeFactorOptimizer:
    def __init__(
        self,
        dopamine=None,
        layer=None,
        pre_layer=None,
        post_layer=None,
        synapse=None,
        lr=1e-3,
        theta_d=0.1,
        theta_p=1.0,
        max_weight=4.0,
        dt=1.0,
        is_feedback=False
    ):
        self.layer = layer
        self.lr = lr
        self.max_weight = max_weight
        self.dt = dt

        self.post_layer = post_layer if post_layer is not None else layer
        self.pre_layer = pre_layer if pre_layer is not None else layer
        self.synapse = (
            synapse if synapse is not None else getattr(layer, "synapse", None)
        )
        if self.synapse is None:
            raise ValueError("Optimizer requires a valid Synapse object.")

        self.n_post = self.synapse.weight.shape[0]
        self.n_pre = self.synapse.weight.shape[1]

        self.theta_d = theta_d
        self.theta_p = theta_p

        self.eligibility_trace = torch.zeros(self.n_post, self.n_pre).to(
            synapse.weight.device
        )

        self.pre_trace = None
        self.post_trace = None

        # STDP parameters
        self.tau_plus = 50.0
        self.tau_minus = 50.0
        self.A_plus = 0.02
        self.A_minus = 0.005

        self.dopamine = dopamine
        self.dopamine_level = 0.0

        self.competition_strength_E = 0.001
        self.competition_strength_I = 0.1

        self.is_feedback = is_feedback

    def step(self, modulation_deltas=None, update_weights=True):
        lr_scale = 1.0
        is_frozen = False
        if modulation_deltas is not None:
            if "lr_scale" in modulation_deltas:
                lr_scale += modulation_deltas["lr_scale"]
            if "learning_inhibition" in modulation_deltas:
                if modulation_deltas["learning_inhibition"] > 0.5:
                    is_frozen = True
        if is_frozen:
            return
        effective_lr = self.lr * lr_scale

        # Calcium
        ca = self.post_layer.calcium.detach().mean(dim=0)
        epsilon = 1e-6
        normalized = (ca - self.theta_d) / (self.theta_p - self.theta_d + epsilon)
        ca_modulation = torch.sigmoid(4 * (normalized - 0.5))
        ca_modulation = ca_modulation.unsqueeze(1)

        # STDP update
        pre_spike = self.pre_layer.post_spike.detach()
        post_spike = self.post_layer.post_spike.detach()
        batch_size = pre_spike.shape[0]
        if self.pre_trace is None or self.pre_trace.shape[0] != batch_size:
            self.pre_trace = torch.zeros_like(pre_spike)
            self.post_trace = torch.zeros_like(post_spike)
        self.pre_trace *= np.exp(-self.dt / self.tau_plus)
        self.post_trace *= np.exp(-self.dt / self.tau_minus)
        self.pre_trace += pre_spike
        self.post_trace += post_spike
        if post_spike is None:
            return
        ltp_matrix = torch.einsum("bi, bj -> ij", post_spike, self.pre_trace)
        ltd_matrix = torch.einsum("bi, bj -> ij", self.post_trace, pre_spike)
        stdp_update = (
            self.A_plus * ltp_matrix - self.A_minus * ltd_matrix
        ) / batch_size

        if not update_weights:
            # eligibility_trace / E(t) = E(t-1) * decay + STDP(t)
            decay = 0.99
            self.eligibility_trace = (
                self.eligibility_trace * decay + ca_modulation * stdp_update
            )
            return

        # effective_dopamine
        self.dopamine_level = (
            self.dopamine.signal() if self.dopamine is not None else 0.0
        )
        
        # Heterosynaptic Competition / Oja's rule
        post_activity = self.post_layer.calcium.mean(dim=0)
        active_mask = (post_activity > self.theta_d).float()
        activity = (active_mask * post_activity).unsqueeze(1)
        excitatory_mask = (self.synapse.weight.data > 0).float()
        inhibitory_mask = (self.synapse.weight.data < 0).float()
        exc_total_strength = torch.sum(
            self.synapse.weight.data * excitatory_mask, dim=1, keepdim=True
        )
        exc_competition = (
            self.competition_strength_E
            * self.synapse.weight.data
            * excitatory_mask
            * exc_total_strength
            * activity
        )
        inh_total_strength = torch.sum(
            torch.abs(self.synapse.weight.data) * inhibitory_mask, dim=1, keepdim=True
        )
        inh_competition = (
            self.competition_strength_I
            * self.synapse.weight.data
            * inhibitory_mask
            * inh_total_strength
            * activity
        )

        if self.is_feedback:
            pre_activity = self.pre_layer.calcium.detach()
            post_error = self.post_layer.calcium.detach()
            error_threshold = 0.1
            active_error = F.relu(post_error - error_threshold)
            excitatory_mask = (self.synapse.weight.data > 0).float()
            inhibitory_mask = (self.synapse.weight.data < 0).float()
            pc_update = torch.einsum("bi, bj -> ij", active_error, pre_activity)
            pc_update_directional = (pc_update * excitatory_mask) - (pc_update * inhibitory_mask)
            weight_decay = 0.001 * self.synapse.weight.data
            delta_w = (self.lr * 0.1) * pc_update_directional - weight_decay
        else:
            trace_scale = 100.0
            # iSTDP
            target_activity = 0.05
            activity_error = (post_activity - target_activity).unsqueeze(1)
            pre_activity = self.pre_trace.mean(dim=0, keepdim=True)
            delta_w_inh_learning = (
                -(self.lr * 0.01) * activity_error * pre_activity * inhibitory_mask
            )

            reward_learning_exc = torch.zeros_like(self.synapse.weight.data)
            # dW = (Learning_Rate * Dopamine * eligibility_trace) - competition_decay
            reward_learning_exc = (
                effective_lr * self.dopamine_level * self.eligibility_trace * trace_scale * excitatory_mask
            )
            decay_term_exc = self.lr * exc_competition
            decay_term_inh = self.lr * inh_competition
            delta_w_exc = reward_learning_exc - decay_term_exc
            delta_w_inh = delta_w_inh_learning - decay_term_inh
            delta_w = delta_w_exc + delta_w_inh
            self.eligibility_trace *= 0.1
        with torch.no_grad():
            self.synapse.weight += delta_w
            self.synapse.enforce_dale_principle()

    def step_structure(self):
        if hasattr(self.synapse, "evolve_connectivity"):
            self.synapse.evolve_connectivity()

    def get_weight_stats(self):
        w = self.synapse.weight.data
        return {
            "mean": w.mean().item(),
            "std": w.std().item(),
            "max": w.max().item(),
            "min": w.min().item(),
            "nonzero": (w != 0).sum().item(),
        }
