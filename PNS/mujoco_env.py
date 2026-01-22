import mujoco
import mujoco.viewer
import numpy as np
import time

class WolfEnv:
    def __init__(self, xml_path):
        """
        初始化環境
        xml_path: 您的 .xml 檔案路徑
        """
        # 1. 載入模型 (Model) - 這是靜態的結構
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        
        # 2. 建立數據 (Data) - 這是動態的狀態 (位置、速度)
        self.data = mujoco.MjData(self.model)
        
        # 準備渲染視窗
        self.viewer = None

    def reset(self):
        """重置環境回到初始狀態"""
        mujoco.mj_resetData(self.model, self.data)
        return self.get_observation()

    def step(self, action):
        """
        核心迴圈：
        1. 接收動作 (action) -> 驅動馬達
        2. 物理運算 (mj_step)
        3. 回傳狀態
        """
        # 將動作傳給馬達 (ctrl)
        # 這裡假設 action 是一個 numpy array
        if action is not None:
            self.data.ctrl[:] = action

        # --- MuJoCo 的核心魔法 ---
        # 讓物理引擎推進一個時間步 (通常是 2ms)
        mujoco.mj_step(self.model, self.data)
        
        return self.get_observation()

    def get_observation(self):
        """獲取當前的感測器數據 (SNN 的輸入)"""
        # 例如：回傳關節角度 (qpos) 和 角速度 (qvel)
        return np.concatenate((self.data.qpos, self.data.qvel))

    def render(self):
        """開啟視窗看一看"""
        if self.viewer is None:
            # 啟動被動視窗
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # 同步畫面
        self.viewer.sync()
        
    def close(self):
        if self.viewer:
            self.viewer.close()