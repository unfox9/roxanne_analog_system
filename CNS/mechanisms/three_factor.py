import torch
import torch.nn as nn

class ThreeFactorOptimizer:
    def __init__(self, layer, n_neurons, lr=1e-3, decay_e=0.95, decay_trace=0.95, max_weight=1.0):
        self.n_neurons = n_neurons
        self.layer = layer
        self.lr = lr
        self.decay_e = decay_e
        self.decay_trace = decay_trace 
        self.max_weight = max_weight
        
        self.eligibility_trace = torch.zeros(self.n_neurons, self.n_neurons).to(layer.synapse.weight.device)
        
        self.pre_trace = torch.zeros(self.n_neurons).to(layer.synapse.weight.device)
        
        self.dopamine_level = 0.0
        self.dopmine_decay = 0.9
        self.dopamine_baseline = 0.0

        self.v_threshold_for_learning = -50.0

        # STDP parameters
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        self.A_plus = 1.0
        self.A_minus = 0.5

    def compute_stdp(self, pre_trace, post_activity):
        #post (N, 1) @ pre (1, N) -> (N, N)
        post_col = post_activity.unsqueeze(1)  
        pre_row = pre_trace.unsqueeze(0)

        hebbian = torch.matmul(post_col, pre_row)

        stdp_update = hebbian * self.A_plus

        return stdp_update

    def step(self, reward):
        current_spikes = self.layer.post_spike.detach()
        current_v = self.layer.v.detach()

        if current_spikes is None:
            return
        
        if self.pre_trace.device != current_spikes.device:
            self.pre_trace = self.pre_trace.to(current_spikes.device)
        if self.eligibility_trace.device != current_spikes.device:
            self.eligibility_trace = self.eligibility_trace.to(current_spikes.device)

        post_excitability = torch.relu(current_v - self.v_threshold_for_learning)
        post_activity = post_excitability.mean(dim=0) # Shape: (N, 1)

        stdp_update = self.compute_stdp(self.pre_trace, post_activity)        
        
        # E(t) = E(t-1) * decay + STDP(t)
        self.eligibility_trace = self.eligibility_trace * self.decay_e + stdp_update
        self.pre_trace = self.pre_trace * self.decay_trace + current_spikes.mean(dim=0)

        self.dopamine_level = self.dopamine_level * self.dopmine_decay + reward
        self.dopamine_level = self.dopamine_level * 0.95 + self.dopamine_baseline * 0.05

        # dW = Learning_Rate * Dopamine * Eligibility
        if abs(self.dopamine_level) > 0.001:
            neuron_types = self.layer.neuron_type.to(self.layer.synapse.weight.device)
            type_modifier = neuron_types.unsqueeze(0)

            delta_w = self.lr * self.dopamine_level * self.eligibility_trace * type_modifier
            
            with torch.no_grad():
                self.layer.synapse.weight += delta_w
                self.layer.enforce_dale_principle(
                    max_exc_weight=self.max_weight, 
                    max_inh_weight=self.max_weight
                )
                
        with torch.no_grad():
            self.layer.synapse.weight.mul_(0.9999)

    def set_reward(self, reward_value):
        self.dopamine_level += reward_value

    def get_dopamine_level(self):
        return self.dopamine_level
    
    def get_weight_stats(self):
        w = self.layer.synapse.weight.data
        return {
            "mean": w.mean().item(),
            "std": w.std().item(),
            "max": w.max().item(),
            "min": w.min().item(),
            "nonzero": (w != 0).sum().item()
        }
                