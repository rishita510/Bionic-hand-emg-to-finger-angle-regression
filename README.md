# 🦾 Bionic Hand — EMG to Finger Angle Regression

A deep learning pipeline that predicts **continuous finger joint angles** from **surface EMG signals** using a CNN-LSTM architecture. Built as part of an internship research project at **BERT Lab, IIT Jodhpur** (Department of Bioscience & Bioengineering).

---

## 📌 Overview

Surface EMG (sEMG) signals captured from forearm muscles are used to estimate real-time finger joint angles — enabling gesture-aware prosthetic/bionic hand control. This project implements the full pipeline from raw signal acquisition to trained regression model.

Inspired by the **TF2AngleNet** architecture (Biomedical Signal Processing and Control, 2025).

---

## 🧰 Hardware Used

| Device | Purpose |
|--------|---------|
| NPG Lite Beast Pack (6-channel BLE) | sEMG signal acquisition |
| Intel RealSense D435i | Depth-based finger tracking |
| MediaPipe | Finger landmark & angle extraction |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Total samples | 95,970 rows |
| EMG channels | 6 (`emg_ch0` – `emg_ch5`) |
| Finger angle targets | 5 |
| Actual EMG sampling rate | ~954 Hz (BLE acquired) |
| Angle update rate | ~9.1 Hz (RealSense + MediaPipe) |

**Finger Angle Targets:**
- `thumb_flex` — Thumb flexion
- `index_mcp` — Index finger MCP joint
- `middle_mcp` — Middle finger MCP joint
- `ring_mcp` — Ring finger MCP joint
- `pinkie_mcp` — Pinky finger MCP joint

---

## ⚙️ Pipeline

```
Raw EMG (954 Hz) + Raw Angles (9.1 Hz)
        ↓
Angle smoothing — Rolling median (window=7)
        ↓
EMG normalization — StandardScaler
        ↓
Sliding window segmentation
   └─ Window size: 200 samples (~210ms)
   └─ Step size: 10 samples
        ↓
CNN-LSTM Model
   └─ CNN: local temporal feature extraction
   └─ LSTM: sequential dependency modeling
        ↓
5 continuous angle predictions
```

---

## 🧠 Model Architecture

```
Input: (batch, 200, 6)
  ↓
Conv1d(6→32, k=5) → BN → ReLU
Conv1d(32→64, k=5) → BN → ReLU
MaxPool1d(2)
  ↓
LSTM(64→128, layers=2, dropout=0.3)
  ↓
Linear(128→64) → ReLU → Dropout(0.3)
Linear(64→5)
  ↓
Output: (batch, 5) — predicted angles
```

**Total parameters:** 251,493

---

## 📈 Results

Trained for **50 epochs** with Adam optimizer and ReduceLROnPlateau scheduler.

| Finger | MAE | R² Score |
|--------|-----|----------|
| Thumb Flexion | 0.0155 | 0.8834 |
| Index MCP | 0.0243 | 0.8968 |
| Middle MCP | 0.0284 | 0.9114 |
| Ring MCP | 0.0271 | 0.8974 |
| Pinky MCP | 0.0206 | 0.9568 |

> All five fingers achieved **R² > 0.88**, with Pinky MCP reaching **0.9568**.

---

## 🚀 Setup & Usage

### Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pandas numpy scikit-learn matplotlib
```

> CUDA 12.4 compatible. Tested on NVIDIA GeForce RTX 2050, PyTorch 2.6.0.

### Run Training

```bash
python cnn_lstm_angle_prediction.py
```

Update `FILE_PATH` in the script to point to your CSV dataset.

### Output
- `cnn_lstm_angle_model.pth` — saved model weights
- `training_curve.png` — loss curve plot

---

## 📁 Project Structure

```
bionic-hand-emg-angle-regression/
│
├── cnn_lstm_angle_prediction.py   # Main training script
├── training_curve.png             # Loss curve
├── cnn_lstm_angle_model.pth       # Saved model weights
└── README.md
```

---

## 🔬 Research Context

This project is part of ongoing research at **BERT Lab, IIT Jodhpur** on EMG-based hand gesture recognition and continuous joint angle estimation for prosthetic hand control.

**Related work:**
- TF2AngleNet — Biomedical Signal Processing and Control (2025)

---

## 👤 Author

**Research Intern — BERT Lab, IIT Jodhpur**
Department of Bioscience & Bioengineering
