from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from CNS.mechanisms.snn_network import IzhikevichLayer
from CNS.mechanisms.snn_synapse import Synapse


@dataclass(frozen=True)
class IzhikevichCPGOutput:
    """Continuous population activity decoded from the spiking CPG."""

    left: float
    right: float

    left_flexor: float
    left_extensor: float
    right_flexor: float
    right_extensor: float

    left_flexor_hz: float
    left_extensor_hz: float
    right_flexor_hz: float
    right_extensor_hz: float


class IzhikevichCPG:
    """Bilateral spiking half-center CPG with explicit inhibitory interneurons.

    Rhythm populations (LF, LE, RF, RE) are excitatory Izhikevich pools.
    Each rhythm pool excites a dedicated inhibitory interneuron pool. Those
    interneurons inhibit the antagonist on the same leg and the same functional
    population on the contralateral leg:

        LF_E -> LF_I -| LE_E, RF_E
        LE_E -> LE_I -| LF_E, RE_E
        RF_E -> RF_I -| RE_E, LF_E
        RE_E -> RE_I -| RF_E, LE_E

    All E->I and I->E interactions use the project's conductance-based Synapse.
    There is deliberately no population-level inhibition matrix and no external
    fatigue state in this first version. Release from inhibition is left to the
    Izhikevich recovery/adaptation dynamics plus synaptic decay.

    Spike trains are low-pass filtered only for readout. The filtered activity
    does not feed back into the neural dynamics.
    """

    POOL_NAMES = ("LF", "LE", "RF", "RE")
    LF, LE, RF, RE = range(4)
    N_POOLS = 4

    def __init__(
        self,
        sim_dt: float = 0.002,
        neural_dt_ms: float = 1.0,
        neurons_per_pool: int = 8,
        inhibitory_neurons_per_pool: int = 4,
        tonic_current: float = 10.0,
        e_to_i_weight: float = 0.16,
        local_i_to_e_weight: float = 0.28,
        contralateral_i_to_e_weight: float = 0.18,
        e_to_i_density: float = 0.75,
        i_to_e_density: float = 0.75,
        excitatory_tau_ms: float = 5.0,
        inhibitory_tau_ms: float = 12.0,
        synaptic_delay_ms: float = 3.0,
        synaptic_g_scale: float = 1.0,
        rate_tau_ms: float = 25.0,
        rate_scale_hz: float = 35.0,
        initial_bias: float = 0.8,
        output_gain: float = 1.0,
        noise_level: float = 0.0,
        seed: int = 1,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        # Deprecated compatibility args from the previous population-current CPG.
        antagonist_inhibition: float | None = None,
        contralateral_inhibition: float | None = None,
        fatigue_gain: float | None = None,
        fatigue_tau_ms: float | None = None,
    ) -> None:
        del antagonist_inhibition, contralateral_inhibition, fatigue_gain, fatigue_tau_ms

        if sim_dt <= 0.0:
            raise ValueError("sim_dt must be > 0")
        if neural_dt_ms <= 0.0:
            raise ValueError("neural_dt_ms must be > 0")
        if neurons_per_pool <= 0 or inhibitory_neurons_per_pool <= 0:
            raise ValueError("pool sizes must be > 0")
        if rate_tau_ms <= 0.0 or rate_scale_hz <= 0.0:
            raise ValueError("rate parameters must be > 0")
        if synaptic_delay_ms < neural_dt_ms:
            raise ValueError("synaptic_delay_ms must be >= neural_dt_ms")

        self.sim_dt = float(sim_dt)
        self.neural_dt_ms = float(neural_dt_ms)
        self.neurons_per_pool = int(neurons_per_pool)
        self.inhibitory_neurons_per_pool = int(inhibitory_neurons_per_pool)
        self.tonic_current = float(tonic_current)
        self.rate_scale_hz = float(rate_scale_hz)
        self.output_gain = float(output_gain)
        self.device = torch.device(device)

        neural_time_per_sim_step_ms = self.sim_dt * 1000.0
        exact_substeps = neural_time_per_sim_step_ms / self.neural_dt_ms
        self.neural_substeps = int(round(exact_substeps))
        if self.neural_substeps < 1 or not math.isclose(
            exact_substeps, self.neural_substeps, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "sim_dt must be an integer multiple of neural_dt_ms / 1000. "
                f"Got sim_dt={self.sim_dt}, neural_dt_ms={self.neural_dt_ms}."
            )

        self._rate_decay = math.exp(-self.neural_dt_ms / float(rate_tau_ms))
        self._spike_trace = torch.zeros(
            self.N_POOLS, dtype=torch.float32, device=self.device
        )
        self._activation = torch.zeros_like(self._spike_trace)

        # LF and RE start slightly favored so the desired diagonal phase is
        # selected without permanently forcing the gait.
        b = float(initial_bias)
        self._startup_bias = torch.tensor(
            [b, -b, -b, b], dtype=torch.float32, device=self.device
        )
        self._startup_bias_decay = math.exp(-self.neural_dt_ms / 150.0)
        self._startup_bias_state = self._startup_bias.clone()

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))

            self.e_pools = [
                IzhikevichLayer(
                    n_neurons=self.neurons_per_pool,
                    synapse=None,
                    exc_ratio=1.0,
                    a_exc=0.02,
                    b_exc=0.20,
                    c_exc=-65.0,
                    d_exc=8.0,
                    dt=self.neural_dt_ms,
                    refractory_steps=2,
                ).to(self.device)
                for _ in range(self.N_POOLS)
            ]

            self.i_pools = [
                IzhikevichLayer(
                    n_neurons=self.inhibitory_neurons_per_pool,
                    synapse=None,
                    exc_ratio=0.0,
                    a_inh=0.02,
                    b_inh=0.25,
                    c_inh=-65.0,
                    d_inh=2.0,
                    dt=self.neural_dt_ms,
                    refractory_steps=2,
                ).to(self.device)
                for _ in range(self.N_POOLS)
            ]

            for layer in [*self.e_pools, *self.i_pools]:
                with torch.no_grad():
                    layer.noise_level.fill_(float(noise_level))
                    layer.neuro_sensitivity.fill_(1.0)

            e_to_i_syn_masks = [
                (torch.rand(self.inhibitory_neurons_per_pool, self.neurons_per_pool,  device=self.device)< e_to_i_density).float()
                for _ in range(self.N_POOLS)
            ]

            i_to_e_syn_masks = [
                (torch.rand(self.neurons_per_pool, self.inhibitory_neurons_per_pool, device=self.device)< i_to_e_density).float()
                for _ in range(self.N_POOLS)
            ]

            # One excitatory synapse per E pool onto its associated I pool.
            self.e_to_i = [
                self._make_synapse(
                    pre=self.e_pools[k],
                    post=self.i_pools[k],
                    density=e_to_i_density,
                    tau_ms=excitatory_tau_ms,
                    max_delay_ms=synaptic_delay_ms,
                    g_scale=synaptic_g_scale,
                    fixed_abs_weight=e_to_i_weight,
                    connection_mask=e_to_i_syn_masks[k]
                )
                for k in range(self.N_POOLS)
            ]

            # Explicit inhibitory pathways reproduce the old topology, but now
            # the sign comes from inhibitory presynaptic neurons + E_inh.
            targets = {
                self.LF: ((self.LE, local_i_to_e_weight),
                          (self.RF, contralateral_i_to_e_weight)),
                self.LE: ((self.LF, local_i_to_e_weight),
                          (self.RE, contralateral_i_to_e_weight)),
                self.RF: ((self.RE, local_i_to_e_weight),
                          (self.LF, contralateral_i_to_e_weight)),
                self.RE: ((self.RF, local_i_to_e_weight),
                          (self.LE, contralateral_i_to_e_weight)),
            }

            self.i_to_e: dict[tuple[int, int], Synapse] = {}
            for source_i, target_list in targets.items():
                for target_e, weight in target_list:
                    self.i_to_e[(source_i, target_e)] = self._make_synapse(
                        pre=self.i_pools[source_i],
                        post=self.e_pools[target_e],
                        density=i_to_e_density,
                        tau_ms=inhibitory_tau_ms,
                        max_delay_ms=synaptic_delay_ms,
                        g_scale=synaptic_g_scale,
                        fixed_abs_weight=weight,
                        connection_mask=i_to_e_syn_masks[source_i]
                    )

        self.reset()



    def _make_synapse(
        self,
        *,
        pre: IzhikevichLayer,
        post: IzhikevichLayer,
        density: float,
        tau_ms: float,
        max_delay_ms: float,
        g_scale: float,
        fixed_abs_weight: float,
        connection_mask: torch.Tensor | None = None,
    ) -> Synapse:
        syn = Synapse(
            pre_layer=pre,
            post_layer=post,
            density=float(density),
            dt=self.neural_dt_ms,
            tau_g=float(tau_ms),
            e_exc=0.0,
            e_inh=-80.0,
            max_delay_ms=float(max_delay_ms),
            max_weight=max(1.0, float(fixed_abs_weight)),
            w_init_multiplier=1.0,
            g_scale=float(g_scale),
            enable_plasticity=False,
            connection_mask=connection_mask,
        ).to(self.device)

        # Keep the random sparse topology, but remove random weight magnitude as
        # a confound for this first CPG experiment.
        with torch.no_grad():
            sign = torch.where(
                syn.neuron_type > 0,
                torch.ones_like(syn.neuron_type),
                -torch.ones_like(syn.neuron_type),
            )
            syn.weight.copy_(
                syn.mask * sign.unsqueeze(0) * float(fixed_abs_weight)
            )
            syn.enforce_dale_principle()
        return syn

    def reset(self) -> None:
        """Reset membrane, synapse, filtered-rate, and startup-phase states."""
        for layer in [*self.e_pools, *self.i_pools]:
            layer.reset_state(batch_size=1, device=self.device)

        for syn in [*self.e_to_i, *self.i_to_e.values()]:
            syn.reset_state(batch_size=1, device=self.device)

        self._spike_trace.zero_()
        self._activation.zero_()
        self._startup_bias_state.copy_(self._startup_bias)

    def _neural_step(self, tonic_scale: float) -> None:
        """Advance all pools synchronously by one neural time step."""
        tonic = self.tonic_current * float(tonic_scale)

        # Snapshot previous spikes first. Every synapse therefore observes the
        # same discrete time slice, independent of Python update order.
        prev_e = [layer.post_spike.clone() for layer in self.e_pools]
        prev_i = [layer.post_spike.clone() for layer in self.i_pools]

        # E -> I currents.
        i_inputs = []
        for k in range(self.N_POOLS):
            i_inputs.append(self.e_to_i[k](prev_e[k], self.i_pools[k].v))

        # Sum all I -> E currents that target each excitatory rhythm pool.
        e_inputs = [
            torch.full(
                (1, self.neurons_per_pool),
                tonic + float(self._startup_bias_state[k]),
                dtype=torch.float32,
                device=self.device,
            )
            for k in range(self.N_POOLS)
        ]
        for (source_i, target_e), syn in self.i_to_e.items():
            e_inputs[target_e] = e_inputs[target_e] + syn(
                prev_i[source_i], self.e_pools[target_e].v
            )

        # Update all neuronal populations only after all currents are computed.
        e_spikes = []
        with torch.no_grad():
            for k in range(self.N_POOLS):
                spikes, _ = self.e_pools[k](e_inputs[k])
                e_spikes.append(spikes)
            for k in range(self.N_POOLS):
                self.i_pools[k](i_inputs[k])

        pool_spikes = torch.stack([s.mean() for s in e_spikes])
        self._spike_trace.mul_(self._rate_decay).add_(
            pool_spikes, alpha=(1.0 - self._rate_decay)
        )

        rate_hz = self._spike_trace * (1000.0 / self.neural_dt_ms)
        self._activation.copy_(
            torch.clamp(rate_hz / self.rate_scale_hz, min=0.0, max=1.0)
        )

        # Bias is only a startup phase selector, not a sustained rhythm driver.
        self._startup_bias_state.mul_(self._startup_bias_decay)

    def step(self, tonic_scale: float = 1.0) -> IzhikevichCPGOutput:
        """Advance one simulation step and return continuous CPG activity."""
        if tonic_scale < 0.0:
            raise ValueError("tonic_scale must be >= 0")

        for _ in range(self.neural_substeps):
            self._neural_step(tonic_scale=tonic_scale)

        a = self._activation.detach().cpu().tolist()
        rate_hz = (
            self._spike_trace * (1000.0 / self.neural_dt_ms)
        ).detach().cpu().tolist()

        lf, le, rf, re = (float(x) for x in a)
        lf_hz, le_hz, rf_hz, re_hz = (float(x) for x in rate_hz)

        left = max(-1.0, min(1.0, self.output_gain * (lf - le)))
        right = max(-1.0, min(1.0, self.output_gain * (rf - re)))

        return IzhikevichCPGOutput(
            left=left,
            right=right,
            left_flexor=lf,
            left_extensor=le,
            right_flexor=rf,
            right_extensor=re,
            left_flexor_hz=lf_hz,
            left_extensor_hz=le_hz,
            right_flexor_hz=rf_hz,
            right_extensor_hz=re_hz,
        )


if __name__ == "__main__":
    cpg = IzhikevichCPG(sim_dt=0.002)

    print("time_s,left,right,LF,LE,RF,RE")
    steps = int(5.0 / cpg.sim_dt)
    print_every = max(1, int(0.1 / cpg.sim_dt))

    for k in range(steps):
        out = cpg.step()
        if k % print_every == 0:
            print(
                f"{k * cpg.sim_dt:.3f},"
                f"{out.left:+.3f},{out.right:+.3f},"
                f"{out.left_flexor:.3f},{out.left_extensor:.3f},"
                f"{out.right_flexor:.3f},{out.right_extensor:.3f}"
            )
