import torch

class HormoneConfig:
    def __init__(self, name, decay_rate=0.99, base_level=0.0):
        self.name = name
        self.level = base_level
        self.decay_rate = decay_rate
        
        self.neuron_effects = {}  
        self.synapse_effects = {} 

    def add_neuron_effect(self, param_name, coefficient):
        self.neuron_effects[param_name] = coefficient

    def add_synapse_effect(self, rule_name, coefficient):
        self.synapse_effects[rule_name] = coefficient

    def update_level(self, influx):
        self.level = self.level * self.decay_rate + influx
        self.level = max(0.0, min(1.0, self.level)) # 限制在 0~1 之間

    def get_effects(self):
        neuron_deltas = {k: v * self.level for k, v in self.neuron_effects.items()}
        synapse_deltas = {k: v * self.level for k, v in self.synapse_effects.items()}
        return neuron_deltas, synapse_deltas


class EndocrineSystem:
    def __init__(self):
        self.hormones = {} 

    def register_hormone(self, hormone_config):
        self.hormones[hormone_config.name] = hormone_config

    def step(self, influx_dict):
        total_neuron_deltas = {}
        total_synapse_deltas = {}

        for name, hormone in self.hormones.items():
            influx = influx_dict.get(name, 0.0)
            hormone.update_level(influx)
            
            n_eff, s_eff = hormone.get_effects()
            
            for param, value in n_eff.items():
                total_neuron_deltas[param] = total_neuron_deltas.get(param, 0.0) + value
                
            for rule, value in s_eff.items():
                total_synapse_deltas[rule] = total_synapse_deltas.get(rule, 0.0) + value

        return total_neuron_deltas, total_synapse_deltas