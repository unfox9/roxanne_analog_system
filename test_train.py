# 初始化 Izhikevich 神經元層和三因子優化器
import torch
from CNS.mechanisms.snn_module import IzhikevichLayer
from CNS.mechanisms.three_factor import ThreeFactorOptimizer
import matplotlib.pyplot as plt


def test_pavlov():
    print("開始巴甫洛夫測試...")
    
    # 1. 設定參數
    n_inputs = 50
    n_outputs = 1  # 單個輸出神經元
    n_steps = 300  # 訓練 200 個回合
    device = 'cpu'
    
    input_gain = 12.0

    layer = IzhikevichLayer(n_inputs, n_outputs, a=0.2, b=0.5)
    # 把學習率調大一點方便觀察
    opt = ThreeFactorOptimizer(layer, n_inputs, n_outputs, lr=0.02, decay_e=0.98, decay_pre=0.8)
    
    # 2. 製作兩個固定的輸入圖案 (Pattern A 和 Pattern B)
    # Pattern A: 前 25 個輸入高機率發火
    pattern_A_prob = torch.zeros(n_inputs)
    pattern_A_prob[:25] = 0.85
    pattern_A_prob[25:] = 0.05
    
    # Pattern B: 後 25 個輸入高機率發火
    pattern_B_prob = torch.zeros(n_inputs)
    pattern_B_prob[:25] = 0.05
    pattern_B_prob[25:] = 0.85
    
    spike_counts = []
    weight_diff = []

    # 3. 訓練迴圈
    for trial in range(n_steps):
        layer.reset_state(batch_size=1, device=device)
        
        # 隨機決定這一回合是給 Pattern A 還是 B
        is_pattern_A = torch.rand(1) > 0.5
        
        if is_pattern_A:
            # 產生符合 A 機率的脈衝
            input_spikes = torch.bernoulli(pattern_A_prob).unsqueeze(0) 
        else:
            # 產生符合 B 機率的脈衝
            input_spikes = torch.bernoulli(pattern_B_prob).unsqueeze(0)
            
        # --- 前向傳播 (模擬 20ms 的時間) ---
        # 為了簡化，我們這裡只跑一個時間步，或者你可以跑一個小迴圈
        total_spikes = 0
        current_input = input_spikes * input_gain
        for t in range(50): # 模擬持續 10ms 的刺激
             # 這裡簡單重複輸入，真實情況應該是動態的
             layer(input_spikes)
             total_spikes += layer.post_spike.sum().item()
             
             # 在每個時間步都要更新 eligibility (但不一定給 reward)
             opt.step(input_spikes, reward=0) 
        
        # --- 計算獎勵 ---
        # 規則：
        # 如果是 A 且發火了 -> 獎勵 (Good!)
        # 如果是 B 且發火了 -> 懲罰 (Bad!)
        # 如果沒發火 -> 沒獎勵 (Neutral)
        
        reward = 0
        if total_spikes > 0:
            if is_pattern_A:
                reward = 1.0  # 獎勵正確的反應
            else:
                reward = -1.0 # 懲罰錯誤的反應
        else:
             # (選用) 懲罰沉默：如果太安靜，稍微懲罰一下，或者不動作
             reward = 0.0
        
        # --- 應用獎勵 (Delayed Reward) ---
        # 在回合結束時，根據累積的 Eligibility Trace 更新權重
        opt.step(input_spikes, reward=reward)

        
        # 紀錄權重變化: 看前25個(A特徵)和後25個(B特徵)的權重差
        w = layer.synapse.weight.data[0]
        diff = w[:25].mean() - w[25:].mean()
        weight_diff.append(diff)
        spike_counts.append(total_spikes)
        if trial % 20 == 0:
            print(f"Trial {trial}: Pattern {'A' if is_pattern_A else 'B'}, Spikes: {total_spikes}, Reward: {reward}, Weight Diff: {diff:.4f}")

    # 4. 繪圖結果
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
    plt.savefig("result_v4.png")
    print("測試完成，結果已保存至 result_v4.png")

if __name__ == "__main__":
    test_pavlov()