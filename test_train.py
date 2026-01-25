# 初始化 Izhikevich 神經元層和三因子優化器
import torch
from CNS.mechanisms.snn_module import IzhikevichLayer
from CNS.mechanisms.three_factor import ThreeFactorOptimizer
import matplotlib.pyplot as plt


def test_pavlov():
    print("開始巴甫洛夫測試...")
    
    n_inputs = 50
    n_outputs = 1  
    n_steps = 1000
    device = torch.device("cuda")
    save_path = "result_v2.png"
    input_gain = 12.0

    layer = IzhikevichLayer(n_inputs, n_outputs, a=0.2, b=0.5)
    layer.to(device)
    opt = ThreeFactorOptimizer(layer, n_inputs, n_outputs, lr=0.02, decay_e=0.98, decay_pre=0.8)
    
    pattern_A_prob = torch.zeros(n_inputs)
    pattern_A_prob[:25] = 0.85
    pattern_A_prob[25:] = 0.05
    pattern_B_prob = torch.zeros(n_inputs)
    pattern_B_prob[:25] = 0.05
    pattern_B_prob[25:] = 0.85
    
    spike_counts = []
    weight_diff = []
 
    for trial in range(n_steps):
        layer.reset_state(batch_size=1, device=device)
        is_pattern_A = torch.rand(1) > 0.5
        if is_pattern_A:
            input_spikes = torch.bernoulli(pattern_A_prob).unsqueeze(0) 
        else:
            input_spikes = torch.bernoulli(pattern_B_prob).unsqueeze(0)
   
        total_spikes = 0
        current_input = input_spikes * input_gain
        input_spikes = input_spikes.to(device)
        for t in range(50): 
             layer(input_spikes)
             total_spikes += layer.post_spike.sum().item()
             opt.step(input_spikes, reward=0) 
        
        reward = 0
        if total_spikes > 0:
            if is_pattern_A:
                reward = 1.0 
            else:
                reward = -1.0
        else:
             reward = 0.0

        opt.step(input_spikes, reward=reward)    
        w = layer.synapse.weight.data[0].cpu().numpy()
        diff = w[:25].mean() - w[25:].mean()
        weight_diff.append(diff)
        spike_counts.append(total_spikes)
        if trial % 20 == 0:
            print(f"Trial {trial}: Pattern {'A' if is_pattern_A else 'B'}, Spikes: {total_spikes}, Reward: {reward}, Weight Diff: {diff:.4f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(weight_diff)
    plt.title("Weight Selectivity (High = Likes A)")
    plt.xlabel("Trials")
    plt.ylabel("W(A) - W(B)")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(spike_counts)
    plt.title("Spike Activity per Trial")
    plt.xlabel("Trials")
    plt.ylabel("Spike Count")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"測試完成，結果已保存至 {save_path}")

if __name__ == "__main__":
    test_pavlov()