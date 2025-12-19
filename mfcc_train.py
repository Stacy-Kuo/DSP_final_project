# train_voice_realtime.py 
import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.model_selection import train_test_split

# ---------- Config ----------
DATA_DIR = "data"
CLASSES = ["shoot", "bomb"]
SR = 16000       # sample rate
DURATION = 1.5   # seconds
N_MFCC = 40
BATCH_SIZE = 16
EPOCHS = 20
DEVICE = torch.device( "cpu") #"cuda" if torch.cuda.is_available() else

# ---------- Preprocessing (新增功能: 訓練前統一長度) ----------
def preprocess_data():
    """
    遍歷所有資料，將音訊長度統一修剪或補零至 SR * DURATION，
    並覆蓋原始檔案。確保訓練時資料形狀絕對一致。
    """
    print("[INFO] Preprocessing audio files (Trimming/Padding)...")
    target_samples = int(SR * DURATION)
    
    processed_count = 0
    for cls in CLASSES:
        class_dir = os.path.join(DATA_DIR, cls)
        if not os.path.exists(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            if not filename.endswith(".wav"):
                continue
            
            filepath = os.path.join(class_dir, filename)
            try:
                # 1. 讀取音訊 (librosa 會自動轉成 float32, range -1~1)
                y, _ = librosa.load(filepath, sr=SR)
                
                # 2. 檢查長度並處理
                if len(y) != target_samples:
                    if len(y) < target_samples:
                        # 太短：補零
                        y = np.pad(y, (0, target_samples - len(y)), 'constant')
                    else:
                        # 太長：裁剪
                        y = y[:target_samples]
                    
                    # 3. 轉回 int16 並儲存 (scipy write 需要 int16 或 float)
                    # 為了跟錄音格式一致，我們轉回 int16
                    y_int16 = np.int16(y * 32767)
                    write(filepath, SR, y_int16)
                    processed_count += 1
            except Exception as e:
                print(f"[WARN] Failed to process {filename}: {e}")
                
    print(f"[INFO] Preprocessing complete. {processed_count} files normalized.")

# ---------- Dataset ----------
class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = SR
        self.duration = DURATION
        self.n_mfcc = N_MFCC
        self.samples = int(SR * DURATION)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 因為已經預處理過，這裡理論上讀進來就是正確長度，
        # 但保留這段邏輯作為雙重保險 (Double Safety)
        y, sr = librosa.load(path, sr=self.sr)
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[:self.samples]
            
        mfcc = librosa.feature.mfcc(y, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=512, hop_length=256)
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
        mfcc = np.expand_dims(mfcc, axis=0).astype(np.float32)
        return torch.tensor(mfcc), torch.tensor(label)

# ---------- CNN Model ----------
class VoiceCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.pool = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU()
        
        # --- 自動計算 Flatten 後的大小 ---
        # 使用真實的 Librosa 模擬運算，確保形狀 100% 匹配
        with torch.no_grad():
            dummy_samples = int(SR * DURATION)
            dummy_audio = np.zeros(dummy_samples)
            dummy_mfcc = librosa.feature.mfcc(y=dummy_audio, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=256)
            dummy_input = torch.tensor(dummy_mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            x = self.relu(self.conv1(dummy_input))
            x = self.pool(self.relu(self.conv2(x)))
            flatten_size = x.view(1, -1).size(1)
        
        print(f"[INFO] Calculated FC1 input size: {flatten_size}")
        
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x

# ---------- Real-time Recording ----------
def record_audio(filename, duration=DURATION, sr=SR):
    print(f"[INFO] Recording {filename} for {duration} seconds...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    write(filename, sr, np.int16(recording * 32767))
    print(f"[INFO] Saved: {filename}")

def collect_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)
    print("[INFO] Press Ctrl+C to stop data collection.")

    try:
        while True:
            cls = input(f"Which class to record ({'/'.join(CLASSES)}): ").strip()
            if cls not in CLASSES:
                print("[WARN] Invalid class.")
                continue
            timestamp = int(time.time()*1000)
            filename = os.path.join(DATA_DIR, cls, f"{cls}_{timestamp}.wav")
            record_audio(filename)
    except KeyboardInterrupt:
        print("\n[INFO] Data collection stopped.")

# ---------- Training ----------
def train_model():
    # *** 步驟 1: 先執行資料前處理 ***
    preprocess_data()
    
    file_paths, labels = [], []
    print("[STEP 1] Loading file paths...")
    for idx, cls in enumerate(CLASSES):
        paths = glob.glob(os.path.join(DATA_DIR, cls, "*.wav"))
        file_paths.extend(paths)
        labels.extend([idx]*len(paths))
    
    if len(file_paths) == 0:
        print("[ERROR] No data found. Please collect audio first.")
        return
    
    print("[STEP 2] Splitting dataset...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)
    
    print("[STEP 3] Creating DataLoaders...")
    train_dataset = VoiceDataset(train_paths, train_labels)
    val_dataset = VoiceDataset(val_paths, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("[STEP 4] Initializing Model...")
    model = VoiceCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("[STEP 5] Starting Training Loop...")
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)
        
        # 避免除以零錯誤 (若 validation set 為空)
        val_acc = correct_val / total_val if total_val > 0 else 0.0

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # save best model
        if val_acc >0.99:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "voice_model.pth")
            print("  >>> Best model saved.")

    print("[INFO] Training completed.")

# ---------- Main ----------
if __name__ == "__main__":
    mode = input("Select mode: [1] collect data, [2] train model: ").strip()
    if mode == "1":
        collect_data()
    elif mode == "2":
        train_model()
    else:
        print("[ERROR] Invalid mode.")