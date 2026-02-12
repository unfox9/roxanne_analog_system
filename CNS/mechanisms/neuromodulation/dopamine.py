import numpy as np


class DopamineSystem:
    def __init__(self, dt=1.0, base_level=0.1):
        self.name = "Dopamine"
        self.dt = dt

        self.base_level = base_level
        self.current_level = base_level

        self.phasic_da = 0.0
        self.tonic_da = base_level

        self.phasic_decay = 0.8
        self.tonic_decay = 0.99

        self.neuron_effects = {}
        self.synapse_effects = {}

        self.add_neuron_effect("input_bias", coefficient=10.0)
        self.add_synapse_effect("lr_scale", coefficient=0.0)

    def add_neuron_effect(self, param_name, coefficient):
        self.neuron_effects[param_name] = coefficient

    def add_synapse_effect(self, rule_name, coefficient):
        self.synapse_effects[rule_name] = coefficient

    def update(self, influx):
        self.phasic_da = self.phasic_da * self.phasic_decay + influx
        self.phasic_da = np.clip(self.phasic_da, -1.0, 1.0)

        self.tonic_da += (self.base_level - self.tonic_da) * (1.0 - self.tonic_decay)

        raw_level = self.tonic_da + self.phasic_da

        self.current_level = np.clip(raw_level, 0.0, 2.0)

        return self.current_level

    def get_deltas(self):
        n_deltas = {k: v * self.current_level for k, v in self.neuron_effects.items()}

        s_deltas = {k: v * self.current_level for k, v in self.synapse_effects.items()}

        return n_deltas, s_deltas

    def signal(self):
        return self.phasic_da
