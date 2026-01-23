import torch
import torch.nn as nn


class ThreeFactorOptimizer:
    def __init__(self, layer, input_size, output_size, lr=1e-3, decay_e=0.95, decay_pre=0.95):

        self.layer = layer
        self.lr = lr
        self.decay_e = decay_e
        self.decay_pre = decay_pre
        self.eligibility_trace = torch.zeros(output_size, input_size).to(layer.synapse.weight.device)
        
        self.pre_trace = torch.zeros(input_size).to(layer.synapse.weight.device)
        self.post_trace_decay = 0.95


    def step(self, input_spikes, reward):
        post_spikes = self.layer.post_spike.detach()

        if input_spikes is None or post_spikes is None:
            return
        
        if self.pre_trace.device != input_spikes.device:
            self.pre_trace = self.pre_trace.to(input_spikes.device)
        if self.eligibility_trace.device != input_spikes.device:
            self.eligibility_trace = self.eligibility_trace.to(input_spikes.device)

        batch_size = input_spikes.shape[0]
        
        if self.pre_trace.device != input_spikes.device:
            self.pre_trace = self.pre_trace.to(input_spikes.device)

        self.pre_trace = self.pre_trace * self.decay_pre + input_spikes.mean(dim=0)
        
        post_activity = post_spikes.mean(dim=0).unsqueeze(1) 
        
        pre_activity = self.pre_trace.unsqueeze(0)
        
        stdp_update = torch.matmul(post_activity, pre_activity)

        # E(t) = E(t-1) * decay + STDP(t)
        self.eligibility_trace = self.eligibility_trace * self.decay_e + stdp_update

        # dW = Learning_Rate * Reward * Eligibility
        if reward != 0:
            delta_w = self.lr * reward * self.eligibility_trace
            
            with torch.no_grad():
                self.layer.synapse.weight += delta_w
                self.layer.synapse.weight.clamp_(0.0, 1.0)