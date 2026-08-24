from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from CNS.mechanisms.snn_network import IzhikevichLayer


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
    """A small bilateral spiking half-center CPG.

    Architecture
    ------------
    Four excitatory Izhikevich populations are used:

        LF = left flexor       LE = left extensor
        RF = right flexor      RE = right extensor

    Within each leg, flexor and extensor populations reciprocally inhibit each
    other at the population-current level.  The same functional populations on
    opposite legs also inhibit each other, encouraging left/right anti-phase
    coordination.

    Spikes are low-pass filtered into a continuous activation signal.  This is
    deliberately analogous to the non-negative flexor/extensor outputs used by
    rate-based CPGs such as Matsuoka oscillators:

        motor_primitive = flexor_activation - extensor_activation

    This class is only the rhythm generator.  It does NOT know about Cassie,
    joints, actuator gear ratios, desired angles, or PD gains.  Those belong in
    the later pattern-formation / gait-decoder layer.

    A slow population fatigue term is enabled by default.  It is a deliberately
    simple Matsuoka-inspired scaffold that makes release from reciprocal
    inhibition robust at locomotion time scales.  Set ``fatigue_gain=0`` later
    if you want to test whether the Izhikevich neurons' own recovery variable
    ``u`` is sufficient for rhythmogenesis.

    Notes on time units
    -------------------
    ``IzhikevichLayer`` uses milliseconds.  ``sim_dt`` is in seconds.  Each
    call to :meth:`step` advances enough 1 ms (by default) neural substeps to
    cover one simulation step.
    """

    # Pool indices.  Keeping them explicit avoids mystery array positions.
    LF = 0
    LE = 1
    RF = 2
    RE = 3
    N_POOLS = 4

    def __init__(
        self,
        sim_dt: float = 0.002,
        neural_dt_ms: float = 1.0,
        neurons_per_pool: int = 8,
        tonic_current: float = 11.0,
        antagonist_inhibition: float = 20.0,
        contralateral_inhibition: float = 10.0,
        fatigue_gain: float = 18.0,
        rate_tau_ms: float = 25.0,
        fatigue_tau_ms: float = 500.0,
        rate_scale_hz: float = 35.0,
        initial_bias: float = 0.8,
        output_gain: float = 1.0,
        noise_level: float = 0.0,
        seed: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        if sim_dt <= 0.0:
            raise ValueError("sim_dt must be > 0")
        if neural_dt_ms <= 0.0:
            raise ValueError("neural_dt_ms must be > 0")
        if neurons_per_pool <= 0:
            raise ValueError("neurons_per_pool must be > 0")
        if rate_tau_ms <= 0.0 or fatigue_tau_ms <= 0.0:
            raise ValueError("time constants must be > 0")
        if rate_scale_hz <= 0.0:
            raise ValueError("rate_scale_hz must be > 0")

        self.sim_dt = float(sim_dt)
        self.neural_dt_ms = float(neural_dt_ms)
        self.neurons_per_pool = int(neurons_per_pool)
        self.tonic_current = float(tonic_current)
        self.fatigue_gain = float(fatigue_gain)
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

        # One layer contains all four populations.  This is substantially
        # cheaper than four tiny PyTorch modules while keeping each pool easy
        # to inspect by reshaping to [batch, 4, neurons_per_pool].
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.layer = IzhikevichLayer(
                n_neurons=self.N_POOLS * self.neurons_per_pool,
                synapse=None,
                exc_ratio=1.0,
                # Regular-spiking neurons.  Their recovery variable u already
                # contributes spike-frequency adaptation.
                a_exc=0.02,
                b_exc=0.20,
                c_exc=-65.0,
                d_exc=8.0,
                dt=self.neural_dt_ms,
                refractory_steps=2,
            ).to(self.device)

        # Start deterministic.  Noise can be reintroduced later when testing
        # robustness rather than basic rhythm generation.
        with torch.no_grad():
            self.layer.noise_level.fill_(float(noise_level))
            self.layer.neuro_sensitivity.fill_(1.0)

        # Inhibition matrix: rows receive inhibition from columns.
        # Desired bilateral pattern is approximately LF <-> RE, then LE <-> RF.
        wa = float(antagonist_inhibition)
        wc = float(contralateral_inhibition)
        self._inhibition = torch.tensor(
            [
                [0.0, wa, wc, 0.0],  # LF inhibited by LE and RF
                [wa, 0.0, 0.0, wc],  # LE inhibited by LF and RE
                [wc, 0.0, 0.0, wa],  # RF inhibited by LF and RE
                [0.0, wc, wa, 0.0],  # RE inhibited by LE and RF
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Break perfect symmetry at startup.  Left flexor and right extensor
        # get a tiny head start, which defines the initial gait phase only.
        b = float(initial_bias)
        self._startup_bias = torch.tensor(
            [b, -b, -b, b], dtype=torch.float32, device=self.device
        )

        self._rate_decay = math.exp(-self.neural_dt_ms / float(rate_tau_ms))
        self._fatigue_decay = math.exp(
            -self.neural_dt_ms / float(fatigue_tau_ms)
        )

        self._spike_trace = torch.zeros(
            self.N_POOLS, dtype=torch.float32, device=self.device
        )
        self._activation = torch.zeros_like(self._spike_trace)
        self._fatigue = torch.zeros_like(self._spike_trace)
        self._input_current = torch.zeros(
            (1, self.N_POOLS, self.neurons_per_pool),
            dtype=torch.float32,
            device=self.device,
        )

        self.reset()

    def reset(self) -> None:
        """Reset membrane states, filtered rates, fatigue, and gait phase."""
        self.layer.reset_state(batch_size=1, device=self.device)
        self._spike_trace.zero_()
        self._activation.zero_()
        self._fatigue.zero_()
        self._input_current.zero_()

    def _neural_step(self, tonic_scale: float) -> None:
        """Advance the spiking network by one neural time step."""
        tonic = self.tonic_current * float(tonic_scale)

        # Matsuoka-like logic, implemented as population currents around real
        # Izhikevich spiking neurons:
        #   drive - antagonist inhibition - contralateral inhibition - fatigue
        group_current = (
            tonic
            + self._startup_bias
            - torch.mv(self._inhibition, self._activation)
            - self.fatigue_gain * self._fatigue
        )

        self._input_current[0, :, :] = group_current[:, None]

        with torch.no_grad():
            spikes, _ = self.layer(
                self._input_current.reshape(1, self.N_POOLS * self.neurons_per_pool)
            )

        pool_spikes = spikes.reshape(
            1, self.N_POOLS, self.neurons_per_pool
        ).mean(dim=2)[0]

        # Exponential synaptic/muscle-like filtering of spike trains.
        self._spike_trace.mul_(self._rate_decay).add_(
            pool_spikes, alpha=(1.0 - self._rate_decay)
        )

        rate_hz = self._spike_trace * (1000.0 / self.neural_dt_ms)
        self._activation.copy_(
            torch.clamp(rate_hz / self.rate_scale_hz, min=0.0, max=1.0)
        )

        # Slow fatigue is the release mechanism for the half-center oscillator.
        self._fatigue.mul_(self._fatigue_decay).add_(
            self._activation, alpha=(1.0 - self._fatigue_decay)
        )

    def step(self, tonic_scale: float = 1.0) -> IzhikevichCPGOutput:
        """Advance one simulation step and return continuous CPG activity.

        Parameters
        ----------
        tonic_scale:
            Multiplicative descending drive.  Keep it at 1.0 for the first
            experiment.  Later it can become a simple speed command.
        """
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

        # This is the key Matsuoka-style translation:
        # antagonist activity becomes one signed motor primitive per leg.
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
    # Tiny smoke test.  Run from the project root with:
    #   python -m CNS.spinalcord.izhikevich_cpg
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