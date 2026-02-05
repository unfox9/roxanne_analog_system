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

    n_neurons1 = 512
    n_neurons2 = 256
    n_neurons3 = 128
    decision_neuron_idx = 0
    n_steps = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "deep_snn.png"

    input_gain = 100.0
    noise_level = 2.0
    learning_rate = 0.001
    max_weight = 1.0

    dopamine = DopamineSystem(dt=1.0, base_level=0.1)

    layer1 = IzhikevichLayer(n_neurons1)
    layer2 = IzhikevichLayer(n_neurons2)
    layer3 = IzhikevichLayer(n_neurons3)

    layer1_recurrent = Synapse(layer=layer1)
    layer2_recurrent = Synapse(layer=layer2)
    layer3_recurrent = Synapse(layer=layer3)
    bridge1_2 = Synapse(pre_layer=layer1, post_layer=layer2)
    bridge2_3 = Synapse(pre_layer=layer2, post_layer=layer3)

    layer1.to(device)
    layer2.to(device)
    layer3.to(device)
    layer1_recurrent.to(device)
    layer2_recurrent.to(device)
    layer3_recurrent.to(device)
    bridge1_2.to(device)
    bridge2_3.to(device)

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

    opt_l1_recurrent = ThreeFactorOptimizer(
        layer=layer1,
        synapse=layer1_recurrent,
        dopamine=dopamine,
        lr=learning_rate,
        max_weight=max_weight,
    )

    opt_bridge1_2 = ThreeFactorOptimizer(
        dopamine=dopamine,
        pre_layer=layer1,
        post_layer=layer2,
        synapse=bridge1_2,
        lr=learning_rate,
        max_weight=max_weight,
    )

    opt_l2_recurrent = ThreeFactorOptimizer(
        layer=layer2,
        synapse=layer2_recurrent,
        dopamine=dopamine,
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

    pattern_A_mask = torch.zeros(n_neurons1)
    pattern_A_mask[:25] = 1.0
    pattern_B_mask = torch.zeros(n_neurons1)
    pattern_B_mask[25:50] = 1.0

    trial_patterns = (torch.rand(n_steps, device=device) > 0.5).float()

    debug_neurons = {
        'layer1': [0, 25, 50], 
        'layer2': [0, 64, 128, 192, 255],
        'layer3': [0, 32, 64, 96, 127] 
    }

    detailed_traces = {
        'trial': [],
        'pattern':[],
        # Layer 1
        'l1_v': {i: [] for i in debug_neurons['layer1']},
        'l1_input': {i: [] for i in debug_neurons['layer1']},
        'l1_spikes': {i: [] for i in debug_neurons['layer1']},
        'l1_calcium': {i: [] for i in debug_neurons['layer1']},
        # Layer 2
        'l2_v': {i: [] for i in debug_neurons['layer2']},
        'l2_input': {i: [] for i in debug_neurons['layer2']},
        'l2_spikes': {i: [] for i in debug_neurons['layer2']},
        'l2_calcium': {i: [] for i in debug_neurons['layer2']},
        # Layer 3
        'l3_v': {i: [] for i in debug_neurons['layer3']},
        'l3_input': {i: [] for i in debug_neurons['layer3']},
        'l3_spikes': {i: [] for i in debug_neurons['layer3']},
        'l3_calcium': {i: [] for i in debug_neurons['layer3']},
        # 突觸統計
        'bridge1_2_current_mean': [],
        'bridge1_2_current_std': [],
        'bridge1_2_active_ratio': [],
        
        'bridge2_3_current_mean': [],
        'bridge2_3_current_std': [],
        'bridge2_3_active_ratio': [],

        'bridge2_3_to_decision': [],

        # 權重
        'bridge2_3_weight_to_decision': [],
        'bridge2_3_to_decision': [],
        'bridge2_3_weight_max': [],
        'bridge2_3_weight_mean': [],
        
        # 多巴胺
        'dopamine_level': [],
        'dopamine_signal': [],
    }

    for trial in range(n_steps):
        is_pattern_A = trial_patterns[trial] == 1.0

        base_input = (pattern_A_mask if is_pattern_A else pattern_B_mask) * input_gain
        base_input = base_input.unsqueeze(0).to(device)

        total_decision_spikes = torch.tensor(0.0, device=device)

        trial_snapshots = []

        for t in range(50):
            noise = torch.randn_like(base_input) * noise_level
            current_input = base_input + noise

            neuron_deltas, synapse_deltas = dopamine.get_deltas()
            spikes1, calcium1 = layer1(
                current_input, neuromodulation_deltas=neuron_deltas
            )
            input_current_2 = bridge1_2(spikes1, layer2.v)
            spikes2, calcium2 = layer2(
                input_current_2, neuromodulation_deltas=neuron_deltas
            )
            input_current_3 = bridge2_3(spikes2, layer3.v)
            spikes3, calcium3 = layer3(
                input_current_3, neuromodulation_deltas=neuron_deltas
            )

            if t % 10 == 0 or t == 49:
                snapshot = {
                    't': t,
                    'l1_v': layer1.v[0, debug_neurons['layer1']].cpu().numpy(),
                    'l1_spikes': spikes1[0, debug_neurons['layer1']].cpu().numpy(),
                    'l2_v': layer2.v[0, debug_neurons['layer2']].cpu().numpy(),
                    'l2_input': input_current_2[0, debug_neurons['layer2']].cpu().numpy(),
                    'l3_v': layer3.v[0, debug_neurons['layer3']].cpu().numpy(),
                    'bridge1_2_current': input_current_2[0].cpu().numpy(),
                }
                trial_snapshots.append(snapshot)

            opt_l1_recurrent.step(modulation_deltas=synapse_deltas)
            opt_bridge1_2.step(modulation_deltas=synapse_deltas)
            opt_l2_recurrent.step(modulation_deltas=synapse_deltas)
            opt_bridge2_3.step(modulation_deltas=synapse_deltas)
            opt_l3_recurrent.step(modulation_deltas=synapse_deltas)

            total_decision_spikes += spikes3[:, decision_neuron_idx].sum()
            
        reward = compute_reward(total_decision_spikes, is_pattern_A)
        dopamine.update(influx=reward)

        if trial % 100 == 0:
            detailed_traces['trial'].append(trial)

            for i in debug_neurons['layer1']:
                detailed_traces['l1_v'][i].append(layer1.v[0, i].item())
                detailed_traces['l1_input'][i].append(layer1.I[0, i].item())
                detailed_traces['l1_spikes'][i].append(spikes1[0, i].item())

            for i in debug_neurons['layer2']:
                detailed_traces['l2_v'][i].append(layer2.v[0, i].item())
                detailed_traces['l2_input'][i].append(layer2.I[0, i].item())
                detailed_traces['l2_spikes'][i].append(spikes2[0, i].item())
                detailed_traces['l2_calcium'][i].append(calcium2[0, i].item())

            for i in debug_neurons['layer3']:
                detailed_traces['l3_v'][i].append(layer3.v[0, i].item())
                detailed_traces['l3_input'][i].append(layer3.I[0, i].item())
                detailed_traces['l3_spikes'][i].append(spikes3[0, i].item())
                detailed_traces['l3_calcium'][i].append(calcium3[0, i].item())

            bridge1_2_currents = input_current_2[0].cpu()
            detailed_traces['bridge1_2_current_mean'].append(bridge1_2_currents.mean().item())
            detailed_traces['bridge1_2_current_std'].append(bridge1_2_currents.std().item())
            detailed_traces['bridge1_2_active_ratio'].append(
                (bridge1_2_currents.abs() > 0.1).float().mean().item()
            )

            bridge2_3_currents = input_current_3[0].cpu()
            detailed_traces['bridge2_3_current_mean'].append(bridge2_3_currents.mean().item())
            detailed_traces['bridge2_3_current_std'].append(bridge2_3_currents.std().item())
            detailed_traces['bridge2_3_active_ratio'].append(
                (bridge2_3_currents.abs() > 0.1).float().mean().item()
            )

            w_to_decision = bridge2_3.weight.data[decision_neuron_idx].cpu().numpy()
            detailed_traces['bridge2_3_weight_to_decision'].append(w_to_decision.copy())
            detailed_traces['bridge2_3_weight_max'].append(
                bridge2_3.weight.data.abs().max().item()
            )
            detailed_traces['bridge2_3_weight_mean'].append(
                bridge2_3.weight.data.abs().mean().item()
            )

            detailed_traces['dopamine_level'].append(dopamine.current_level)
            detailed_traces['dopamine_signal'].append(dopamine.signal())

            print(f"\n=== Trial {trial} ===")
            print(f"Pattern: {'A' if is_pattern_A else 'B'}")
            print(f"Decision Spikes: {total_decision_spikes:.0f}")

            print(f"\n--- Layer 1 ---")
            print(f"Layer 1 Input Current:")
            print(f"  Mean: {layer1.I.mean():.2f}")
            print(f"  Std: {layer1.I.std():.2f}")
            print(f"  Active ratio: {(layer1.I.abs() > 0.1).float().mean():.2%}")
            print(f"\nLayer 1 Voltages (selected):")
            for i in debug_neurons['layer1']:
                print(f"  Neuron {i}: {layer1.v[0, i]:.2f}")

            print(f"\n--- Layer 2 ---")
            print(f"Layer 2 Input Current:")
            print(f"  Mean: {layer2.I.mean():.2f}")
            print(f"  Std: {layer2.I.std():.2f}")
            print(f"  Active ratio: {(layer2.I.abs() > 0.1).float().mean():.2%}")
            print(f"\nLayer 2 Voltages (selected):")
            for i in debug_neurons['layer2']:
                print(f"  Neuron {i}: {layer2.v[0, i]:.2f}")

            print(f"\n--- Layer 3 ---")
            print(f"Layer 3 Input Current:")
            print(f"  Mean: {layer3.I.mean():.2f}")
            print(f"  Std: {layer3.I.std():.2f}")
            print(f"  Active ratio: {(layer3.I.abs() > 0.1).float().mean():.2%}")
            print(f"\nLayer 3 Voltages (selected):")
            for i in debug_neurons['layer3']:
                print(f"  Neuron {i}: {layer3.v[0, i]:.2f}")
            

    plot_detailed_debug(detailed_traces, debug_neurons, save_path=save_path)


def plot_detailed_debug(traces, debug_neurons, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    trials = traces['trial']
    
    # Row 1: 神經元電壓軌跡
    for idx, neuron_id in enumerate(debug_neurons['layer1'][:3]):
        axes[0, idx].plot(trials, traces['l1_v'][neuron_id])
        axes[0, idx].set_title(f'Layer 1 Neuron {neuron_id} Voltage')
        axes[0, idx].set_xlabel('Trial')
        axes[0, idx].set_ylabel('Voltage (mV)')
        axes[0, idx].grid(True, alpha=0.3)
    
    # Row 2: Layer 2 輸入電流與電壓
    axes[1, 0].plot(trials, traces['bridge1_2_current_mean'], label='Mean')
    axes[1, 0].fill_between(
        trials,
        np.array(traces['bridge1_2_current_mean']) - np.array(traces['bridge1_2_current_std']),
        np.array(traces['bridge1_2_current_mean']) + np.array(traces['bridge1_2_current_std']),
        alpha=0.3
    )
    axes[1, 0].set_title('Bridge 1→2 Current Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(trials, traces['bridge1_2_active_ratio'])
    axes[1, 1].set_title('Active Synapse Ratio (1→2)')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Layer 2 代表性神經元
    for neuron_id in debug_neurons['layer2']:
        axes[1, 2].plot(trials, traces['l2_v'][neuron_id], label=f'N{neuron_id}')
    axes[1, 2].set_title('Layer 2 Selected Neurons Voltage')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Row 3: 權重演化
    weight_history = np.array(traces['bridge2_3_weight_to_decision'])
    # 選幾個代表性的突觸
    sample_synapses = [0, 10, 50, 100, 128]
    for syn_id in sample_synapses:
        if syn_id < weight_history.shape[1]:
            axes[2, 0].plot(trials, weight_history[:, syn_id], label=f'Syn {syn_id}')
    axes[2, 0].set_title('Weights to Decision Neuron')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 權重分布熱圖
    im = axes[2, 1].imshow(weight_history.T, aspect='auto', cmap='RdBu_r', 
                           interpolation='nearest')
    axes[2, 1].set_title('Weight Evolution Heatmap')
    axes[2, 1].set_xlabel('Trial (sampled)')
    axes[2, 1].set_ylabel('Synapse ID')
    plt.colorbar(im, ax=axes[2, 1])
    
    # 最終權重分布直方圖
    final_weights = weight_history[-1]
    axes[2, 2].hist(final_weights, bins=30, edgecolor='black', alpha=0.7)
    axes[2, 2].axvline(x=0, color='r', linestyle='--')
    axes[2, 2].set_title('Final Weight Distribution')
    axes[2, 2].set_xlabel('Weight Value')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Debug traces saved to {save_path}")


def compute_reward(spikes, is_patternA):
    fired = spikes > 0
    if is_patternA:
        return 1.0 if fired else -0.3
    else:
        return -1.0 if fired else 0.3


if __name__ == "__main__":
    test_pavlov_unified()
