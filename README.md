# DSP_final_project
# Multimodal Biosignals Space-Invader Game

An interactive **Space-Invader–style game** controlled by multiple human biosignals and perceptual modalities, including **ECG, EMG, speech (MFCC), and computer vision–based hand tracking**. This project integrates classical DSP techniques taught in class with independently designed signal processing and real-time interaction modules.

---

## 🎮 Project Overview

This project transforms physiological and behavioral signals into intuitive game controls:

* **Computer Vision (CV)** – Continuous left/right spaceship movement using hand tracking
* **EMG (Electromyography)** – Voluntary muscle activation for firing actions
* **Speech Recognition (MFCC + CNN)** – Discrete voice commands (e.g., *shoot*, *bomb*)
* **ECG (Electrocardiography)** – Heart-rate–driven adaptive game difficulty

The goal is not only functional correctness, but also **robust real-time performance**, **low latency**, and **engaging biofeedback-driven interaction**.

---

## 🧠 System Architecture

The system follows a modular real-time architecture:

```
Sensors / Inputs
 ├─ ECG + EMG  ──► SensorThread (DSP processing)
 ├─ Microphone ──► MFCC + CNN (VoiceControl)
 ├─ Webcam     ──► Mediapipe Hand Tracking
 └─ Keyboard   ──► Fallback Control

            ▼
      Control Signals
 (movement, firing, commands, difficulty)
            ▼
        Pygame Engine
 (game logic, rendering, collision)
```

All modules operate concurrently and communicate with a central **Pygame-based game loop** running at 60 FPS.

---

## 🛠 Signal Processing Pipelines

### ECG (Course Content)

* Instrumentation amplifier + analog band-limiting (conceptual front-end)
* Digital band-pass filtering (5–40 Hz)
* Squaring + moving-window integration
* Adaptive thresholding for R-peak detection
* BPM estimation

**Game mapping:** Higher BPM → faster enemy movement and descent (biofeedback loop)

---

### EMG (Independent Work)

* Band-pass filtering (20–150 Hz)
* Full-wave rectification
* Envelope extraction via low-pass smoothing
* Rolling mean normalization
* Adaptive threshold + refractory logic

**Design choice:** No 60 Hz notch filter, preserving broadband muscle activation energy and temporal fidelity.

**Game mapping:** Muscle contraction → firing trigger

---

### Speech Recognition (Course Content + Extension)

* Short-time framing + Hamming window
* STFT → Mel filter banks → MFCC extraction
* CNN-based command classification (*shoot*, *bomb*)
* Sliding window + energy gating for real-time robustness

---

### Computer Vision (Independent Work)

* Mediapipe hand landmark detection
* Palm center estimation
* Screen-coordinate mapping
* Exponential moving average (EMA) smoothing

**Game mapping:** Horizontal spaceship movement

---

## 📁 Project Structure

```
├─ main.py                # Main game loop (Pygame)
├─ sensor_thread.py       # ECG & EMG acquisition + processing
├─ emg_processor.py       # EMG DSP pipeline 
├─ ecg_processor.py       # ECG DSP pipeline 
├─ voice_control.py       # MFCC + CNN speech recognition
├─ mfcc_train.py          # using MFCC + CNN to train model
├─ README.md
```

---

## ▶️ How to Run

### Requirements

* Python 3.8+
* numpy, scipy
* pygame
* opencv-python
* mediapipe
* torch, torchaudio

Install dependencies:

```bash
pip install numpy scipy pygame opencv-python mediapipe torch torchaudio
```

### Run the Game

```bash
python main.py
```

If sensors or camera are unavailable, the system automatically falls back to keyboard control.

---

## 🎯 Controls Summary

| Modality  | Function                  |
| --------- | ------------------------- |
| Hand (CV) | Move spaceship left/right |
| EMG       | Fire bullet               |
| Voice     | Shoot / Bomb              |
| ECG       | Modulate difficulty       |
| Keyboard  | Backup control            |

---

## ✨ Key Contributions

* Designed an **adaptive EMG processing pipeline** robust to user variability
* Integrated **real-time multimodal biosignals** into a game engine
* Demonstrated **biofeedback-driven difficulty modulation** using ECG
* Combined DSP, machine learning, and HCI principles in a single system

---

## 📌 Notes

* ECG and MFCC pipelines follow DSP methods taught in class
* EMG processing, CV control, multimodal interaction logic, and system integration are independently implemented
* The project emphasizes **engineering trade-offs** over purely offline accuracy

---

## 👤 Author

Chih-Ling Kuo
Department of Electrical Engineering

---

*This project demonstrates how classical DSP techniques can be transformed into playful, real-time human–machine interaction systems.*
