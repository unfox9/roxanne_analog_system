from PNS.mujoco_env import WolfEnv
import time
import numpy as np

# 1. 指向您的 XML
xml_path = "configs/agility_cassie/scene.xml"  # 或是 "configs/hello_world.xml"

# 2. 建立環境
env = WolfEnv(xml_path)

# 3. 測試迴圈
print("模擬開始！按 Ctrl+C 停止")
try:
    while True:
        # 假裝這是從大腦傳來的訊號 (全零，或是隨機亂動)
        dummy_action = np.zeros(env.model.nu) # nu = number of actuators (馬達數量)
        
        # 走一步
        env.step(dummy_action)
        
        # 渲染畫面
        env.render()
        
        # 控制更新頻率 (不然會跑太快)
        time.sleep(0.002)

except KeyboardInterrupt:
    env.close()
    print("模擬結束")