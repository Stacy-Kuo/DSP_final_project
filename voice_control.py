import torch
import torch.nn as nn
import torchaudio
import numpy as np
import sounddevice as sd
from collections import deque
from scipy.special import softmax

# ---------- Config ----------
SR = 16000
DURATION = 1.5
N_MFCC = 40
WINDOW_SEC = 1.5       
STEP_SEC = 0.2
DEVICE = "cpu"  
THRESHOLD = 0.5       
LABELS = ["shoot", "bomb"]  
ENERGY_THRESHOLD = 0.005   
MIN_ACTIVE_RATIO = 0.9   

# [新增 Config] 冷卻時間：辨識成功後，休息幾秒不重複辨識
COOLDOWN_SEC = 1.0  

# ---------- CNN Model (保持不變) ----------
class VoiceCNN(nn.Module):
    def __init__(self, n_classes=len(LABELS)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.pool = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU()
        
        with torch.no_grad():
            dummy_audio = np.zeros(int(SR*DURATION))
            dummy_mfcc = torchaudio.transforms.MFCC(
                sample_rate=SR,
                n_mfcc=N_MFCC,
                melkwargs={"n_fft":512, "hop_length":256, "n_mels":40}
            )(torch.tensor(dummy_audio, dtype=torch.float32).unsqueeze(0))
            dummy_input = dummy_mfcc.unsqueeze(0)
            x = self.relu(self.conv1(dummy_input))
            x = self.pool(self.relu(self.conv2(x)))
            flatten_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- Real-time Voice Recognition ----------
class VoiceControl:
    def __init__(self, model_path="voice_model.pth"):
        self.device = DEVICE
        self.model = VoiceCNN().to(self.device)
        # 嘗試載入模型，如果沒有檔案也不要崩潰 (方便測試)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except:
            print(f"Warning: {model_path} not found. Using random weights.")
            
        self.model.eval()
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SR,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft":512, "hop_length":256, "n_mels":40}
        )
        
        self.window_len = int(WINDOW_SEC * SR)
        self.step_len = int(STEP_SEC * SR)
        self.buffer = deque(maxlen=self.window_len)
        self.current_command = None
        
        # [新增] 冷卻計數器
        self.cooldown_steps = int(COOLDOWN_SEC / STEP_SEC) # 需要跳過的步數
        self.current_cooldown = 0
        
        self.stream = sd.InputStream(
            channels=1,
            samplerate=SR,
            callback=self.audio_callback,
            blocksize=self.step_len
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        # 處理 Input Overflow
        # 如果 status 存在，我們檢查是否為 input overflow。
        # 如果是 overflow，我們選擇忽略它不 print，避免洗版。
        if status:
            if "input overflow" not in str(status).lower():
                print("[voice_control] Stream warning:", status)
            # else: 就是 overflow，我們靜默略過

        audio = indata[:, 0]
        self.buffer.extend(audio)

        if len(self.buffer) < self.window_len:
            return

        #  冷卻機制 (Debouncing)
        # 如果還在冷卻時間內，直接跳過辨識，但 buffer 還是要收音以保持時間連續
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
            return

        window_data = np.array(self.buffer)[-self.window_len:]

        # ===== 1️聲音能量檢查（VAD）=====
        rms = np.sqrt(np.mean(window_data ** 2))

        if rms < ENERGY_THRESHOLD:
            self.current_command = None
            return

        # ===== 2 進 CNN =====
        mfcc = self.mfcc_transform(
            torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
        )
        mfcc = mfcc.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(mfcc)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            max_prob = probs.max()
            if max_prob >= THRESHOLD:
                cmd_idx = np.argmax(probs)
                self.current_command = LABELS[cmd_idx]

                # 辨識成功後，設定冷卻時間
                # 接下來的幾次 callback 將不會進行預測，防止同一個字被重複抓取
                self.current_cooldown = self.cooldown_steps 
                
                # 選項：辨識成功後是否清空 Buffer？
                # 這裡不清空，依靠冷卻時間滑過去即可
            else:
                self.current_command = None

    def get_command(self):
        """回傳即時辨識結果: shoot / bomb / None"""
        cmd = self.current_command
        self.current_command = None 
        return cmd

    def stop(self):
        self.stream.stop()
        self.stream.close()

# ---------- Example Usage ----------
if __name__ == "__main__":
    vc = VoiceControl("voice_model.pth")
    print(f"Listening... (Cooldown: {COOLDOWN_SEC}s) Press Ctrl+C to stop.")
    try:
        while True:
            cmd = vc.get_command()
            if cmd:
                print(f"[COMMAND DETECTED] {cmd}")
            # 加上一點點 sleep 減少 CPU 佔用
            import time
            time.sleep(0.01)
    except KeyboardInterrupt:
        vc.stop()
        print("\nStopped.")