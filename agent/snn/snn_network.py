import torch
import torch.nn as nn

class IzhikevichLayer(nn.Module):
    def __init__(self, input_size, output_size, a=0.02, b=0.2, c=-65.0, d=8.0):
        super().__init__()
        # 突觸連接 (Linear Layer)
        self.synapse = nn.Linear(input_size, output_size, bias=False)
        
        # Izhikevich 參數 (可以是單一數值，也可以是向量以支援不同神經元)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # 狀態變數
        self.v = None # 膜電位
        self.u = None # 恢復變數 (Recovery Variable)
        self.trace = None # 資格跡 (給 Three-factor 用)
        
        # 記錄脈衝
        self.pre_spike = None
        self.post_spike = None
        
        # Trace 的衰減率
        self.trace_decay = 0.95

    def reset_state(self, batch_size, device):
        # 初始化 v 為 -65mV (靜止電位), u 為 b*v
        self.v = torch.ones(batch_size, self.synapse.out_features).to(device) * -65.0
        self.u = self.v * self.b
        self.trace = torch.zeros_like(self.v)

    def forward(self, x):
        # 1. 計算輸入電流 I (來自突觸權重)
        # x 是上一層的 Spikes (0 或 1)
        self.pre_spike = x.detach()
        I = self.synapse(x)
        
        # 初始化狀態 (如果還沒初始化的話)
        if self.v is None:
            self.reset_state(x.shape[0], x.device)
            
        # --- Izhikevich 數值積分 (Euler Method, dt=1ms) ---
        # 這裡我們假設 dt=1 簡化計算，很多論文這樣做以節省算力
        
        # v' = 0.04v^2 + 5v + 140 - u + I
        v_next = self.v + (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        
        # u' = a(bv - u)
        u_next = self.u + self.a * (self.b * self.v - self.u)
        
        # 更新狀態
        self.v = v_next
        self.u = u_next
        
        # --- 發火與重置機制 ---
        # 判斷是否發火 (Threshold 通常設為 30mV)
        self.post_spike = (self.v >= 30.0).float()
        
        # 重置邏輯 (Vectorized Reset)
        # 如果發火(1): v -> c, u -> u + d
        # 如果沒發火(0): v -> v, u -> u
        
        # v = (1 - spike) * v + spike * c
        self.v = (1.0 - self.post_spike) * self.v + self.post_spike * self.c
        
        # u = u + spike * d
        self.u = self.u + self.post_spike * self.d
        
        # --- 更新資格跡 (給 Three-factor Learning) ---
        self.trace = self.trace * self.trace_decay + self.post_spike
        
        return self.post_spike