# sensor_thread.py
import threading
import time
import serial
from ecg_processor import ECGProcessor
from emg_processor import EMGProcessor

class SensorThread(threading.Thread):
    def __init__(self, port="COM3", baud=115200):
        super().__init__()
        self.daemon = True
        
        # default values
        self.bpm = 0
        self.ecg_value = 0.0       # 用來控制下降速度
        self.emg_fire = False      # 射擊事件 trigger

        self.ecg = ECGProcessor() 
        self.emg = EMGProcessor()

        try:
            self.ser = serial.Serial(port, baud, timeout=0.01)
            self.connected = True
        except:
            print("[SensorThread] No COM detected. Using default values.")
            self.connected = False

    def run(self):
        while True:
            if not self.connected:
                time.sleep(0.02)
                self.emg_fire = False
                self.ecg_value = 0.0
                self.bpm = 0.0
                continue

            try:
                line = self.ser.readline().decode().strip()
                # 假設訊號格式: ECG_value, EMG_value
                if "," in line:
                    ecg_raw, emg_raw = line.split(",")
                    ecg_raw = float(ecg_raw)
                    emg_raw = float(emg_raw)

                    # === 即時處理 ===
                    self.bpm, self.ecg_value = self.ecg.update(ecg_raw)     # 回傳心跳頻率 or normalized 值
                    self.emg_fire = self.emg.update(emg_raw)
                    if self.emg.update(emg_raw):
                        self.emg_fire = True
                        time.sleep(0.2)
                    
         
            except:
                continue
