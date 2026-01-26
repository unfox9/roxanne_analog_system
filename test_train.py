import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from CNS.mechanisms.snn_module import IzhikevichLayer
from CNS.mechanisms.three_factor import ThreeFactorOptimizer


def test_pavlov_unified():
    print("開始巴甫洛夫測試...")
    
    n_neurons = 100
    decision_neuron_idx = 99 
    n_steps = 1000
    device = torch.device("cuda") 
    save_path = "result_unified_fixed.png"
    
    # === 關鍵參數調整 ===
    input_gain = 30.0  # 降低輸入強度（從 20 降到 15）
    noise_level = 2.0  # 降低噪音
    learning_rate = 0.01 # 大幅降低學習率（從 0.05 降到 0.005）
    max_weight = 4.0  # 設定權重上限

    # 初始化層（加入 refractory period）
    layer = IzhikevichLayer(
        n_neurons, 
        d=2.0,
        refractory_steps=5 # 5 個時間步的不反應期
    ) 
    
    # === 關鍵：降低連接機率和初始權重 ===
    layer.init_weights(
        strategy="random",
        connection_prob=0.5,  # 只有 10% 的連接
        max_weight=5.0 # 初始權重最大 0.3
    ) 
    layer.to(device)
    
    # 只在最開始重置一次
    layer.reset_state(batch_size=1, device=device)
    
    # 初始化優化器
    opt = ThreeFactorOptimizer(
        layer, 
        n_neurons, 
        theta_d=0.3,
        theta_p=0.8,
        lr=learning_rate,  # 使用更小的學習率
        max_weight=max_weight  # 傳入權重上限
    )

    # 定義輸入模式
    pattern_A_mask = torch.zeros(n_neurons)
    pattern_A_mask[:25] = 1.0
    
    pattern_B_mask = torch.zeros(n_neurons)
    pattern_B_mask[25:50] = 1.0
    
    # 記錄數據
    spike_counts = []
    weight_diff = []
    rewards_history = []
    weight_max_history = []  # 追蹤權重最大值
 
    for trial in range(n_steps):
        # 隨機選擇模式
        is_pattern_A = torch.rand(1) > 0.5
        
        # 準備基礎輸入
        if is_pattern_A:
            base_input = (pattern_A_mask * input_gain).unsqueeze(0).to(device)
        else:
            base_input = (pattern_B_mask * input_gain).unsqueeze(0).to(device)
   
        total_decision_spikes = 0
        
        # 時間步演化
        for t in range(50): 
            # 每個時間步都加新的噪音（但強度較低）
            noise = torch.randn_like(base_input) * noise_level
            noise[:, decision_neuron_idx] += 5.0

            current_input = base_input + noise
            
            # 前向傳播
            spikes, calcium = layer(current_input)
            
            # 記錄決策神經元的放電
            decision_spike = layer.post_spike[:, decision_neuron_idx].item()
            total_decision_spikes += decision_spike
            
            # 每步都更新 eligibility（無獎勵）
            opt.step(reward=0.0)  

        # 根據放電和模式給獎勵
        reward = 0
        if total_decision_spikes > 0:
            if is_pattern_A:
                reward = 1.0 
            else:
                reward = -1.0 
        else:
            reward = 0.0

        # 用累積的 eligibility 和獎勵學習
        opt.step(reward=reward)   
        
        # 記錄權重差異和最大權重
        w = layer.synapse.weight.data[decision_neuron_idx].cpu().numpy()
        diff = w[:25].mean() - w[25:50].mean()
        w_max = layer.synapse.weight.data.abs().max().item()
        
        weight_diff.append(diff)
        spike_counts.append(total_decision_spikes)
        rewards_history.append(reward)
        weight_max_history.append(w_max)
        
        if trial % 100 == 0:
            recent_avg = sum(spike_counts[-10:]) / 10 if len(spike_counts) >= 10 else total_decision_spikes
            stats = opt.get_weight_stats()
            print(f"Trial {trial}: Pattern {'A' if is_pattern_A else 'B'}, "
                  f"Spikes: {total_decision_spikes:.0f}, "
                  f"Avg: {recent_avg:.1f}, "
                  f"Reward: {reward:+.1f}, "
                  f"W_diff: {diff:.4f}, "
                  f"W_max: {w_max:.4f}")

    # === 改進的可視化 ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 權重選擇性
    axes[0, 0].plot(weight_diff, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title("Weight Selectivity (Should → Positive)")
    axes[0, 0].set_xlabel("Trials")
    axes[0, 0].set_ylabel("W(A) - W(B)")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 放電活動
    axes[0, 1].plot(spike_counts, alpha=0.7)
    axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Max (bad)')
    axes[0, 1].axhline(y=5, color='g', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].set_title("Spike Activity (Should be < 50)")
    axes[0, 1].set_xlabel("Trials")
    axes[0, 1].set_ylabel("Spike Count")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 權重最大值追蹤
    axes[0, 2].plot(weight_max_history, alpha=0.7)
    axes[0, 2].axhline(y=max_weight, color='r', linestyle='--', 
                       label=f'Limit: {max_weight}')
    axes[0, 2].set_title("Max Weight (Should Stay ≤ Limit)")
    axes[0, 2].set_xlabel("Trials")
    axes[0, 2].set_ylabel("Max |Weight|")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 獎勵歷史
    axes[1, 0].plot(rewards_history, alpha=0.3, color='gray')
    # 移動平均
    window = 50
    if len(rewards_history) >= window:
        moving_avg = [sum(rewards_history[i:i+window])/window 
                      for i in range(len(rewards_history)-window)]
        axes[1, 0].plot(range(window, len(rewards_history)), moving_avg, 
                        color='red', linewidth=2, label='Moving Avg (50)')
        axes[1, 0].legend()
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title("Reward Signal")
    axes[1, 0].set_xlabel("Trials")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 放電分佈直方圖
    axes[1, 1].hist(spike_counts, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=50, color='r', linestyle='--', 
                       label='Problematic (always firing)')
    axes[1, 1].set_title("Spike Count Distribution")
    axes[1, 1].set_xlabel("Spike Count")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 多巴胺水平（如果有記錄）
    axes[1, 2].text(0.5, 0.5, 
                    f"Final Stats:\n\n"
                    f"Avg Spikes: {sum(spike_counts)/len(spike_counts):.2f}\n"
                    f"Spike Std: {torch.tensor(spike_counts).float().std():.2f}\n"
                    f"Final W_diff: {weight_diff[-1]:.4f}\n"
                    f"Max Weight: {max(weight_max_history):.4f}\n"
                    f"Positive Rewards: {sum(1 for r in rewards_history if r > 0)}\n"
                    f"Negative Rewards: {sum(1 for r in rewards_history if r < 0)}",
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    axes[1, 2].set_title("Summary")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n結果已儲存至 {save_path}")
    
    # === 詳細統計分析 ===
    print("\n" + "="*50)
    print("統計分析")
    print("="*50)
    print(f"平均放電次數: {sum(spike_counts)/len(spike_counts):.2f}")
    print(f"放電次數標準差: {torch.tensor(spike_counts).float().std():.2f}")
    print(f"最大放電次數: {max(spike_counts)}")
    print(f"最小放電次數: {min(spike_counts)}")
    print(f"\n最終權重差異: {weight_diff[-1]:.4f}")
    print(f"權重差異範圍: [{min(weight_diff):.4f}, {max(weight_diff):.4f}]")
    print(f"\n最大權重值: {max(weight_max_history):.4f} (限制: {max_weight})")
    print(f"\n總正獎勵: {sum(1 for r in rewards_history if r > 0)}")
    print(f"總負獎勵: {sum(1 for r in rewards_history if r < 0)}")
    print(f"總零獎勵: {sum(1 for r in rewards_history if r == 0)}")
    
    # 檢查是否有問題
    if max(spike_counts) == 50:
        print("\n⚠️  警告：仍有神經元每步都在放電 (50/50)")
        print("   建議：進一步降低 input_gain 或增加 refractory_steps")
    if max(weight_max_history) > max_weight * 1.1:
        print(f"\n⚠️  警告：權重超過限制 ({max(weight_max_history):.4f} > {max_weight})")
        print("   建議：檢查 enforce_dale_principle() 是否正確執行")


if __name__ == "__main__":
    test_pavlov_unified()