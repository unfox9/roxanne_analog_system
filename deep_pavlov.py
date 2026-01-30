import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from CNS.mechanisms.snn_network import IzhikevichLayer
from CNS.mechanisms.three_factor import ThreeFactorOptimizer
from CNS.mechanisms.neuromodulation.dopamine import DopamineSystem
from CNS.mechanisms.snn_synapse import Synapse


def test_pavlov_unified():
    print("開始巴甫洛夫測試(Deep)...")

    n_neurons1 = 2048
    n_neurons2 = 1024
    n_neurons3 = 512
    decision_neuron_idx = 0
    n_steps = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "result_unified.png"

    input_gain = 10.0
    noise_level = 10.0
    learning_rate = 0.001
    max_weight = 4.0

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

    spike_counts = []
    rewards_history = []
    weight_max_history = []

    for trial in range(n_steps):
        is_pattern_A = trial_patterns[trial] == 1.0

        base_input = (pattern_A_mask if is_pattern_A else pattern_B_mask) * input_gain
        base_input = base_input.unsqueeze(0).to(device)

        total_decision_spikes = torch.tensor(0.0, device=device)

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
            opt_l1_recurrent.step(modulation_deltas=synapse_deltas)
            opt_bridge1_2.step(modulation_deltas=synapse_deltas)
            opt_l2_recurrent.step(modulation_deltas=synapse_deltas)
            opt_bridge2_3.step(modulation_deltas=synapse_deltas)
            opt_l3_recurrent.step(modulation_deltas=synapse_deltas)
            total_decision_spikes += spikes3[:, decision_neuron_idx].sum()
            reward = compute_reward(total_decision_spikes, is_pattern_A)
            dopamine.update(influx=reward)

        spike_counts.append(total_decision_spikes.item())
        rewards_history.append(reward)

        if trial % 100 == 0:
            v_mean = layer2.v.mean().item()
            input_mean = input_current_2.mean().item()
            print(
                f"DEBUG: L2 Input Current: {input_mean:.2f}, L2 Voltage: {v_mean:.2f}"
            )

            w = bridge2_3.weight.data[decision_neuron_idx].cpu().numpy()
            w_max = bridge2_3.weight.data.abs().max().item()

            weight_max_history.append(w_max)
            recent_avg = (
                sum(spike_counts[-10:]) / 10
                if len(spike_counts) >= 10
                else total_decision_spikes
            )
            stats = opt_l3_recurrent.get_weight_stats()
            print(
                f"Trial {trial}: Pattern {'A' if is_pattern_A else 'B'}, "
                f"Spikes: {total_decision_spikes:.0f}, "
                f"Avg: {recent_avg:.1f}, "
                f"Reward: {reward:+.1f}, "
                f"W_max: {w_max:.4f}, "
                f"Dopamine level:{dopamine.current_level:.4f}, "
                f"Signal:{dopamine.signal():.4f}"
            )

    # === 改進的可視化 ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 權重選擇性
    axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Weight Selectivity (Should → Positive)")
    axes[0, 0].set_xlabel("Trials")
    axes[0, 0].set_ylabel("W(A) - W(B)")
    axes[0, 0].grid(True, alpha=0.3)

    # 放電活動
    axes[0, 1].plot(spike_counts, alpha=0.7)
    axes[0, 1].axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Max (bad)")
    axes[0, 1].axhline(y=5, color="g", linestyle="--", alpha=0.5, label="Target")
    axes[0, 1].set_title("Spike Activity (Should be < 50)")
    axes[0, 1].set_xlabel("Trials")
    axes[0, 1].set_ylabel("Spike Count")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 權重最大值追蹤
    axes[0, 2].plot(weight_max_history, alpha=0.7)
    axes[0, 2].axhline(
        y=max_weight, color="r", linestyle="--", label=f"Limit: {max_weight}"
    )
    axes[0, 2].set_title("Max Weight (Should Stay ≤ Limit)")
    axes[0, 2].set_xlabel("Trials")
    axes[0, 2].set_ylabel("Max |Weight|")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 獎勵歷史
    axes[1, 0].plot(rewards_history, alpha=0.3, color="gray")
    # 移動平均
    window = 50
    if len(rewards_history) >= window:
        moving_avg = [
            sum(rewards_history[i : i + window]) / window
            for i in range(len(rewards_history) - window)
        ]
        axes[1, 0].plot(
            range(window, len(rewards_history)),
            moving_avg,
            color="red",
            linewidth=2,
            label="Moving Avg (50)",
        )
        axes[1, 0].legend()
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 0].set_title("Reward Signal")
    axes[1, 0].set_xlabel("Trials")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].grid(True, alpha=0.3)

    # 放電分佈直方圖
    axes[1, 1].hist(spike_counts, bins=20, edgecolor="black", alpha=0.7)
    axes[1, 1].axvline(
        x=50, color="r", linestyle="--", label="Problematic (always firing)"
    )
    axes[1, 1].set_title("Spike Count Distribution")
    axes[1, 1].set_xlabel("Spike Count")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 多巴胺水平（如果有記錄）
    axes[1, 2].text(
        0.5,
        0.5,
        f"Final Stats:\n\n"
        f"Avg Spikes: {sum(spike_counts)/len(spike_counts):.2f}\n"
        f"Spike Std: {torch.tensor(spike_counts).float().std():.2f}\n"
        f"Max Weight: {max(weight_max_history):.4f}\n"
        f"Positive Rewards: {sum(1 for r in rewards_history if r > 0)}\n"
        f"Negative Rewards: {sum(1 for r in rewards_history if r < 0)}",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Summary")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n結果已儲存至 {save_path}")

    # === 詳細統計分析 ===
    print("\n" + "=" * 50)
    print("統計分析")
    print("=" * 50)
    print(f"平均放電次數: {sum(spike_counts)/len(spike_counts):.2f}")
    print(f"放電次數標準差: {torch.tensor(spike_counts).float().std():.2f}")
    print(f"最大放電次數: {max(spike_counts)}")
    print(f"最小放電次數: {min(spike_counts)}")
    print(f"\n最大權重值: {max(weight_max_history):.4f} (限制: {max_weight})")
    print(f"\n總正獎勵: {sum(1 for r in rewards_history if r > 0)}")
    print(f"總負獎勵: {sum(1 for r in rewards_history if r < 0)}")
    print(f"總零獎勵: {sum(1 for r in rewards_history if r == 0)}")

    # 檢查是否有問題
    if max(spike_counts) == 50:
        print("\n⚠️  警告：仍有神經元每步都在放電 (50/50)")
        print("   建議：進一步降低 input_gain 或增加 refractory_steps")
    if max(weight_max_history) > max_weight * 1.1:
        print(
            f"\n⚠️  警告：權重超過限制 ({max(weight_max_history):.4f} > {max_weight})"
        )
        print("   建議：檢查 enforce_dale_principle() 是否正確執行")


def compute_reward(spikes, is_patternA):
    fired = spikes > 0
    if is_patternA:
        return 1.0 if fired else -0.3
    else:
        return -1.0 if fired else 0.3


if __name__ == "__main__":
    test_pavlov_unified()
