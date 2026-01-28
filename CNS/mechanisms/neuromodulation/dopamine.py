import numpy as np

class DopamineSystem:
    def __init__(self, dt=1.0, decay_ms=20.0, base_level=0.1):
        self.name = "Dopamine"
        self.dt = dt

        self.decay_factor = np.exp(-dt / decay_ms)
        self.base_level = base_level
        
        self.current_level = base_level
        
        self.neuron_effects = {}
        self.synapse_effects = {}
        
        self.add_neuron_effect('input_bias', coefficient=5.0)
        
        self.add_synapse_effect('lr_scale', coefficient=5.0)

    def add_neuron_effect(self, param_name, coefficient):
        self.neuron_effects[param_name] = coefficient

    def add_synapse_effect(self, rule_name, coefficient):
        self.synapse_effects[rule_name] = coefficient

    def update(self, influx):
        self.current_level = (
            self.base_level + 
            (self.current_level - self.base_level) * self.decay_factor + 
            influx
        )
        
        self.current_level = np.clip(self.current_level, 0.0, 5.0) # 允許暫時衝高超過 1.0

        return self.current_level

    def get_deltas(self):
        n_deltas = {k: v * self.current_level for k, v in self.neuron_effects.items()}
        
        s_deltas = {k: v * self.current_level for k, v in self.synapse_effects.items()}
        
        return n_deltas, s_deltas

    def signal(self):
        return self.current_level - self.base_level