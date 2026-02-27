import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from CNS.mechanisms.snn_network import IzhikevichLayer
from CNS.mechanisms.three_factor import ThreeFactorOptimizer
from CNS.mechanisms.neuromodulation.dopamine import DopamineSystem
from CNS.mechanisms.snn_synapse import Synapse
import numpy as np


def test_pavlov_unified():
    print("開始巴甫洛夫測試(Deep)...")

    n_neurons1 = 256
    n_neurons2 = 1024
    n_neurons3 = 64
    n_steps = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "deep_snn.png"

    input_gain = 10.0
    learning_rate = 0.01
    max_weight = 2.0

    dopamine = DopamineSystem(dt=1.0, base_level=0.1)

    layer1 = IzhikevichLayer(n_neurons1, enable_alic=False)
    layer2 = IzhikevichLayer(n_neurons2)
    layer3 = IzhikevichLayer(n_neurons3)

    density = 0.4
    layer1_recurrent = Synapse(layer=layer1, density=density)
    layer2_recurrent = Synapse(layer=layer2, density=0.3)
    layer3_recurrent = Synapse(layer=layer3, density=0.8)
    bridge1_2 = Synapse(pre_layer=layer1, post_layer=layer2, density=density)
    bridge2_3 = Synapse(pre_layer=layer2, post_layer=layer3, density=0.6)
    bridge2_1 = Synapse(pre_layer=layer2, post_layer=layer1, density=0.8)
    bridge3_2 = Synapse(pre_layer=layer3, post_layer=layer2, density=0.8)

    layer1.to(device)
    layer2.to(device)
    layer3.to(device)
    layer1_recurrent.to(device)
    layer2_recurrent.to(device)
    layer3_recurrent.to(device)
    bridge1_2.to(device)
    bridge2_3.to(device)
    bridge2_1.to(device)
    bridge3_2.to(device)

    layer1.synapse = layer1_recurrent
    layer2.synapse = layer2_recurrent
    layer3.synapse = layer3_recurrent

    layer1.reset_state(batch_size=1, device=device)
    layer2.reset_state(batch_size=1, device=device)
    layer3.reset_state(batch_size=1, device=device)
    layer1_recurrent.reset_state(batch_size=1, device=device)
    layer2_recurrent.reset_state(batch_size=1, device=device)
    layer3_recurrent.reset_state(batch_size=1, device=device)
    bridge1_2.reset_state(batch_size=1, device=device)
    bridge2_3.reset_state(batch_size=1, device=device)
    bridge2_1.reset_state(batch_size=1, device=device)
    bridge3_2.reset_state(batch_size=1, device=device)

    opt_l1_recurrent = ThreeFactorOptimizer(
        layer=layer1,
        synapse=layer1_recurrent,
        lr=learning_rate,
        max_weight=max_weight,
    )

    opt_bridge1_2 = ThreeFactorOptimizer(
        pre_layer=layer1,
        post_layer=layer2,
        synapse=bridge1_2,
        lr=learning_rate,
        max_weight=max_weight,
    )

    opt_l2_recurrent = ThreeFactorOptimizer(
        layer=layer2,
        synapse=layer2_recurrent,
        lr=learning_rate,
        max_weight=max_weight,
    )

    opt_bridge2_3 = ThreeFactorOptimizer(
        dopamine=dopamine,
        pre_layer=layer2,
        post_layer=layer3,
        synapse=bridge2_3,
        lr=learning_rate,
        max_weight=max_weight,
    )

    opt_l3_recurrent = ThreeFactorOptimizer(
        layer=layer3,
        synapse=layer3_recurrent,
        dopamine=dopamine,
        lr=learning_rate,
        max_weight=max_weight,
    )

    n_exc_l1 = int(n_neurons1 * 0.8)
    n_exc_l2 = int(n_neurons2 * 0.8)
    n_exc_l3 = int(n_neurons3 * 0.8)
    pattern_A_mask = torch.zeros(n_neurons1)
    pattern_B_mask = torch.zeros(n_neurons1)
    pattern_A_mask[: n_exc_l1 // 2] = 1.0
    pattern_B_mask[n_exc_l1 // 2 : n_exc_l1] = 1.0

    group_A = slice(0, n_exc_l3 // 2)
    group_B = slice(n_exc_l3 // 2, n_exc_l3)

    trial_patterns = (torch.rand(n_steps, device=device) > 0.5).float()

    correct_history = []

    debug_neurons = {
        "layer1": np.linspace(0, n_neurons1 - 1, 3, dtype=int).tolist(),
        "layer2": np.linspace(0, n_neurons2 - 1, 5, dtype=int).tolist(),
        "layer3": np.linspace(0, n_neurons3 - 1, 5, dtype=int).tolist(),
    }

    detailed_traces = {
        "trial": [],
        "pattern": [],
        # Layer 1
        "l1_v": {i: [] for i in debug_neurons["layer1"]},
        "l1_input": {i: [] for i in debug_neurons["layer1"]},
        "l1_spikes": {i: [] for i in debug_neurons["layer1"]},
        "l1_calcium": {i: [] for i in debug_neurons["layer1"]},
        # Layer 2
        "l2_v": {i: [] for i in debug_neurons["layer2"]},
        "l2_input": {i: [] for i in debug_neurons["layer2"]},
        "l2_spikes": {i: [] for i in debug_neurons["layer2"]},
        "l2_calcium": {i: [] for i in debug_neurons["layer2"]},
        # Layer 3
        "l3_v": {i: [] for i in debug_neurons["layer3"]},
        "l3_input": {i: [] for i in debug_neurons["layer3"]},
        "l3_spikes": {i: [] for i in debug_neurons["layer3"]},
        "l3_calcium": {i: [] for i in debug_neurons["layer3"]},
        # 突觸統計
        "bridge1_2_current_mean": [],
        "bridge1_2_current_std": [],
        "bridge1_2_active_ratio": [],
        "bridge2_3_current_mean": [],
        "bridge2_3_current_std": [],
        "bridge2_3_active_ratio": [],
        # 權重
        "weight_to_groupA_mean": [],
        "weight_to_groupB_mean": [],
        "weight_to_groupA_std": [],
        "weight_to_groupB_std": [],
        "bridge2_3_to_decision": [],
        "bridge2_3_weight_max": [],
        "bridge2_3_weight_mean": [],
        # 多巴胺
        "dopamine_level": [],
        "dopamine_signal": [],
    }

    for trial in range(n_steps):
        is_pattern_A = trial_patterns[trial] == 1.0

        base_input = (pattern_A_mask if is_pattern_A else pattern_B_mask) * input_gain
        base_input = base_input.unsqueeze(0).to(device)

        total_decision_spikes = torch.tensor(0.0, device=device)
        total_spikes_A = torch.tensor(0.0, device=device)
        total_spikes_B = torch.tensor(0.0, device=device)

        trial_snapshots = []

        spikes2_prev = torch.zeros(1, n_neurons2, device=device)
        spikes3_prev = torch.zeros(1, n_neurons3, device=device)

        for t in range(50):
            current_input = base_input

            neuron_deltas, synapse_deltas = dopamine.get_deltas()

            fb_1 = bridge2_1(spikes2_prev, layer1.v)
            fb_2 = bridge3_2(spikes3_prev, layer2.v)

            spikes1, calcium1 = layer1(
                current_input + fb_1, neuromodulation_deltas=neuron_deltas
            )
            input_current_2 = bridge1_2(spikes1, layer2.v)
            spikes2, calcium2 = layer2(
                input_current_2 + fb_2, neuromodulation_deltas=neuron_deltas
            )
            input_current_3 = bridge2_3(spikes2, layer3.v)
            spikes3, calcium3 = layer3(
                input_current_3, neuromodulation_deltas=neuron_deltas
            )

            spikes2_prev = spikes2.detach()
            spikes3_prev = spikes3.detach()

            if t % 10 == 0 or t == 49:
                snapshot = {
                    "t": t,
                    "l1_v": layer1.v[0, debug_neurons["layer1"]].cpu().numpy(),
                    "l1_spikes": spikes1[0, debug_neurons["layer1"]].cpu().numpy(),
                    "l2_v": layer2.v[0, debug_neurons["layer2"]].cpu().numpy(),
                    "l2_input": input_current_2[0, debug_neurons["layer2"]]
                    .cpu()
                    .numpy(),
                    "l3_v": layer3.v[0, debug_neurons["layer3"]].cpu().numpy(),
                    "bridge1_2_current": input_current_2[0].cpu().numpy(),
                }
                trial_snapshots.append(snapshot)

            opt_l1_recurrent.step(update_weights=False)
            opt_bridge1_2.step(update_weights=False)
            opt_l2_recurrent.step(update_weights=False)
            opt_bridge2_3.step(update_weights=False)
            opt_l3_recurrent.step(update_weights=False)

            total_spikes_A += spikes3[:, group_A].sum()
            total_spikes_B += spikes3[:, group_B].sum()
            total_decision_spikes += spikes3.sum()

        rate_A, rate_B, winner, corret, reward = compute_reward(
            total_spikes_A, total_spikes_B, is_pattern_A, n_exc_l3
        )
        correct_history.append(corret)
        dopamine.update(influx=reward)
        da_signal = dopamine.signal()
        for opt in [opt_l1_recurrent, opt_bridge1_2, opt_l2_recurrent]:
            opt.step()
        for opt in [opt_bridge2_3, opt_l3_recurrent]:
            opt.step(da_override=da_signal, modulation_deltas=synapse_deltas)

        if trial % 100 == 0:
            detailed_traces["trial"].append(trial)

            for i in debug_neurons["layer1"]:
                detailed_traces["l1_v"][i].append(layer1.v[0, i].item())
                detailed_traces["l1_input"][i].append(layer1.I[0, i].item())
                detailed_traces["l1_spikes"][i].append(spikes1[0, i].item())

            for i in debug_neurons["layer2"]:
                detailed_traces["l2_v"][i].append(layer2.v[0, i].item())
                detailed_traces["l2_input"][i].append(layer2.I[0, i].item())
                detailed_traces["l2_spikes"][i].append(spikes2[0, i].item())
                detailed_traces["l2_calcium"][i].append(calcium2[0, i].item())

            for i in debug_neurons["layer3"]:
                detailed_traces["l3_v"][i].append(layer3.v[0, i].item())
                detailed_traces["l3_input"][i].append(layer3.I[0, i].item())
                detailed_traces["l3_spikes"][i].append(spikes3[0, i].item())
                detailed_traces["l3_calcium"][i].append(calcium3[0, i].item())

            bridge1_2_currents = input_current_2[0].cpu()
            detailed_traces["bridge1_2_current_mean"].append(
                bridge1_2_currents.mean().item()
            )
            detailed_traces["bridge1_2_current_std"].append(
                bridge1_2_currents.std().item()
            )
            detailed_traces["bridge1_2_active_ratio"].append(
                (bridge1_2_currents.abs() > 0.1).float().mean().item()
            )

            bridge2_3_currents = input_current_3[0].cpu()
            detailed_traces["bridge2_3_current_mean"].append(
                bridge2_3_currents.mean().item()
            )
            detailed_traces["bridge2_3_current_std"].append(
                bridge2_3_currents.std().item()
            )
            detailed_traces["bridge2_3_active_ratio"].append(
                (bridge2_3_currents.abs() > 0.1).float().mean().item()
            )

            w_to_groupA = bridge2_3.weight.data[group_A, :n_exc_l2]
            w_to_groupB = bridge2_3.weight.data[group_B, :n_exc_l2]
            detailed_traces["weight_to_groupA_mean"].append(w_to_groupA.mean().item())
            detailed_traces["weight_to_groupB_mean"].append(w_to_groupB.mean().item())
            detailed_traces["weight_to_groupA_std"].append(w_to_groupA.std().item())
            detailed_traces["weight_to_groupB_std"].append(w_to_groupB.std().item())
            detailed_traces["bridge2_3_weight_max"].append(
                bridge2_3.weight.data.abs().max().item()
            )
            detailed_traces["bridge2_3_weight_mean"].append(
                bridge2_3.weight.data.abs().mean().item()
            )

            detailed_traces["dopamine_level"].append(dopamine.current_level)
            detailed_traces["dopamine_signal"].append(dopamine.signal())

            print(f"\n=== Trial {trial} ===")
            print(f"Pattern: {'A' if is_pattern_A else 'B'}")
            print(f"Decision Spikes: {total_decision_spikes:.0f}")
            print(f"Group A rate: {rate_A:.4f} | Group B rate: {rate_B:.4f}")
            print(f"Winner: {winner} | Correct: {corret}")

            print(f"\n--- Layer 1 ---")
            print(f"Layer 1 Input Current:")
            print(f"  Mean: {layer1.I.mean():.2f}")
            print(f"  Std: {layer1.I.std():.2f}")
            print(f"  Active ratio: {(layer1.I.abs() > 0.1).float().mean():.2%}")
            print(f"\nLayer 1 Voltages (selected):")
            for i in debug_neurons["layer1"]:
                print(f"  Neuron {i}: {layer1.v[0, i]:.2f}")

            print(f"\n--- Layer 2 ---")
            print(f"Layer 2 Input Current:")
            print(f"  Mean: {layer2.I.mean():.2f}")
            print(f"  Std: {layer2.I.std():.2f}")
            print(f"  Active ratio: {(layer2.I.abs() > 0.1).float().mean():.2%}")
            print(f"\nLayer 2 Voltages (selected):")
            for i in debug_neurons["layer2"]:
                print(f"  Neuron {i}: {layer2.v[0, i]:.2f}")

            print(f"\n--- Layer 3 ---")
            print(f"Layer 3 Input Current:")
            print(f"  Mean: {layer3.I.mean():.2f}")
            print(f"  Std: {layer3.I.std():.2f}")
            print(f"  Active ratio: {(layer3.I.abs() > 0.1).float().mean():.2%}")
            print(f"\nLayer 3 Voltages (selected):")
            for i in debug_neurons["layer3"]:
                print(f"  Neuron {i}: {layer3.v[0, i]:.2f}")

            print(f"\n--- Dopamine ---")
            print(f"  Dopamine level:{dopamine.current_level:.4f}")
            print(f"  Signal:{dopamine.signal():.4f}")

            print(f"\n--- Weight ---")
            w = bridge2_3.weight.data
            print(f"exc→groupA: {w[group_A, :n_exc_l2].mean():.4f}")
            print(f"exc→groupB: {w[group_B, :n_exc_l2].mean():.4f}")
            print(f"inh→groupA: {w[group_A, n_exc_l2:].mean():.4f}")
            print(f"inh→groupB: {w[group_B, n_exc_l2:].mean():.4f}")

    print("\n\n--- Final Results ---")
    print(f"Total Trials: {n_steps}")
    total_accuracy = sum(correct_history) / len(correct_history)
    print(f"Overall Accuracy: {total_accuracy:.2%}")
    recent_accuracy = sum(correct_history[-100:]) / 100.0
    print(f"Final 100 Trials Accuracy: {recent_accuracy:.2%}")
    plot_detailed_debug(detailed_traces, debug_neurons, save_path=save_path)


def plot_detailed_debug(traces, debug_neurons, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    trials = traces["trial"]

    # Row 1: 神經元電壓軌跡
    for idx, neuron_id in enumerate(debug_neurons["layer1"][:3]):
        axes[0, idx].plot(trials, traces["l1_v"][neuron_id])
        axes[0, idx].set_title(f"Layer 1 Neuron {neuron_id} Voltage")
        axes[0, idx].set_xlabel("Trial")
        axes[0, idx].set_ylabel("Voltage (mV)")
        axes[0, idx].grid(True, alpha=0.3)

    # Row 2: Layer 2 輸入電流與電壓
    axes[1, 0].plot(trials, traces["bridge1_2_current_mean"], label="Mean")
    axes[1, 0].fill_between(
        trials,
        np.array(traces["bridge1_2_current_mean"])
        - np.array(traces["bridge1_2_current_std"]),
        np.array(traces["bridge1_2_current_mean"])
        + np.array(traces["bridge1_2_current_std"]),
        alpha=0.3,
    )
    axes[1, 0].set_title("Bridge 1→2 Current Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(trials, traces["bridge1_2_active_ratio"])
    axes[1, 1].set_title("Active Synapse Ratio (1→2)")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].grid(True, alpha=0.3)

    # Layer 2 代表性神經元
    for neuron_id in debug_neurons["layer2"]:
        axes[1, 2].plot(trials, traces["l2_v"][neuron_id], label=f"N{neuron_id}")
    axes[1, 2].set_title("Layer 2 Selected Neurons Voltage")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # axes[2, 0] - Group A vs B 的 mean 權重演化
    axes[2, 0].plot(
        trials,
        traces["weight_to_groupA_std"],
        color="purple",
        alpha=0.5,
        label="Group A Std",
    )
    axes[2, 0].plot(
        trials,
        traces["weight_to_groupB_std"],
        color="orange",
        alpha=0.5,
        label="Group B Std",
    )
    axes[2, 0].plot(
        trials, traces["weight_to_groupA_mean"], color="red", label="Group A Mean"
    )
    axes[2, 0].plot(
        trials, traces["weight_to_groupB_mean"], color="blue", label="Group B Mean"
    )
    axes[2, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2, 0].set_title("Bridge2→3 Weight: Group A vs B")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # axes[2, 1] - 兩者的差值，更直觀看分化
    diff = np.array(traces["weight_to_groupA_mean"]) - np.array(
        traces["weight_to_groupB_mean"]
    )
    axes[2, 1].plot(trials, diff, color="purple")
    axes[2, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2, 1].set_title("Group A - Group B (Differentiation)")
    axes[2, 1].grid(True, alpha=0.3)

    # axes[2, 2] - dopamine 信號，確認 reward 是否正常
    axes[2, 2].plot(trials, traces["dopamine_level"], label="level")
    axes[2, 2].plot(trials, traces["dopamine_signal"], label="signal")
    axes[2, 2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2, 2].set_title("Dopamine")
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Debug traces saved to {save_path}")


def compute_reward(spikes_A, spikes_B, is_patternA, n_exc):
    rate_A = spikes_A.item() / (n_exc // 2)
    rate_B = spikes_B.item() / (n_exc // 2)

    if rate_A == rate_B:
        return rate_A, rate_B, "Tie", False, -0.1

    a_wins = rate_A > rate_B
    correct = a_wins == is_patternA
    margin = abs(rate_A - rate_B)
    winner = "A" if a_wins else "B"
    reward = min(1.0, 0.3 + margin * 10) if correct else max(-1.0, -0.3 - margin * 10)

    return rate_A, rate_B, winner, correct, reward


if __name__ == "__main__":
    test_pavlov_unified()
