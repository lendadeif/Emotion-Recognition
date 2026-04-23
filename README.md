# 🧠 Smart Room AI System
<img width="1280" height="853" alt="smart home" src="https://github.com/user-attachments/assets/ae554c31-2c7c-4215-9a08-dda22c74f60f" />


Real-time facial analysis system combining **emotion recognition** and **drowsiness detection** to automatically control room lighting — running on Raspberry Pi with a camera only.

Detects 5 emotions: **Angry, Happy, Neutral, Sad, Surprise** — finetuned MobilenetV2 on balanced FER dataset with **69% accuracy**.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Raspberry Pi                     │
│                                                      │
│   ┌──────────┐    ┌───────────────────┐              │
│   │  Camera  │───▶│     AI Engine     │              │
│   └──────────┘    │                   │              │
│                   │  • Emotion CNN    │───▶  Lights  │
│                   │  • EAR / PERCLOS  │              │
│                   │  • Head Pose      │              │
│                   └───────────────────┘              │
└──────────────────────────────────────────────────────┘
```

### Decision Logic

| Detected State     | Lights          | Temperature | Reason                   |
|--------------------|-----------------|-------------|--------------------------|
| Sleeping           | 10% warm dim    | Warm        | Do not disturb           |
| Drowsy             | 30% warm soft   | Warm        | Gentle, easy on the eyes |
| Angry / Sad        | 45–60% warm     | Warm        | Calming environment      |
| Happy / Neutral    | 75–80% natural  | Neutral     | Maintain comfort         |
| Surprise           | 90% cool bright | Cool        | Alert / focus mode       |

---

## 📋 Requirements

### Software
- Python 3.9+
- Raspberry Pi OS (64-bit recommended)
- Webcam or Raspberry Pi Camera Module v2

### Hardware
- Raspberry Pi 4 (2GB+ RAM recommended)
- Camera module or USB webcam
- Smart RGB bulb (Philips Hue or similar) **or** relay + standard bulb

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/lendadeif/Emotion-Recognition.git
cd emotion-recognition
```

### 2. Create a Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## 🚀 Run

```bash
python main.py
```

| Key   | Action                    |
|-------|---------------------------|
| `Q`   | Quit                      |
| `R`   | Recalibrate EAR threshold |
| `ESC` | Quit                      |

---

## 📁 Project Structure

```
emotion-recognition/
├── main.py                  # Entry point — AI loop + light control
├── sleep_detection.py       # EAR, PERCLOS, head-pose drowsiness logic
├── emotion_engine.py        # CNN inference + smoothing
├── lights.py                # Smart light controller
├── model/
│   └── FER_B.keras          # Trained CNN weights
├── requirements.txt
└── README.md
```

---

## 📦 requirements.txt

```
tensorflow>=2.12.0
opencv-python>=4.7.0
numpy>=1.23.0
mediapipe>=0.10.0
scipy>=1.10.0
requests>=2.28.0
```

---

## 🧠 Emotion Model

| Property      | Value                                    |
|---------------|------------------------------------------|
| Architecture  | Custom CNN (from scratch)                |
| Input size    | 48×48 grayscale                          |
| Classes       | 5 — Angry, Happy, Neutral, Sad, Surprise |
| Val Accuracy  | 69%                                      |
| Macro F1      | 0.69                                     |
| Format        | `.keras`                                 |
| Smoothing     | Majority vote over last 10 frames        |


---

## 😴 Drowsiness Detection

Built on **MediaPipe Face Landmarker** (468 landmarks) with three fused signals:

| Signal     | Weight | Description                                  |
|------------|--------|----------------------------------------------|
| EAR        | 50%    | Eye Aspect Ratio — eye openness per frame    |
| PERCLOS    | 35%    | % of frames eyes closed in last 3 sec window |
| Head pitch | 15%    | Forward head nod angle                       |

**Auto-calibration** runs at startup — measures your personal open-eye EAR baseline and sets the threshold at 60% of it, adapting to any face shape or lighting condition.

Drowsy score thresholds: `> 0.75` → DROWSY, `> 0.92` → SLEEPING.

---

## 💡 Smart Light Integration

### Philips Hue
Set your bridge IP and API key in `lights.py`:
```python
HUE_BRIDGE_IP = "192.168.x.x"
HUE_API_KEY   = "your-api-key"
LIGHT_ID      = "1"
```

### No Smart Bulb
If no smart bulb is connected, the system simulates light changes as a real-time tinted brightness overlay on the camera feed so you can still see the system working.

---

## 📊 Live HUD

```
AWAKE / DROWSY / SLEEPING         ← sleep state
EAR: 0.312  thr: 0.187            ← current vs calibrated threshold
L: 0.318  R: 0.306  diff: 0.012  ← per-eye + asymmetry guard
PERCLOS: 12%  closed: 3f          ← rolling closed-eye %
Pitch: 4.2°                       ← head nod angle
[Drowsy score bar]

Emotion: Happy                    ← smoothed emotion label

──────────────────────────────────
LIGHT: 80%  [NEUTRAL]             ← current light output
Happy — natural balanced light    ← reason
[Brightness bar]
```

---

## ⚠️ Notes

- On Raspberry Pi 4, emotion inference runs every 3rd frame to maintain ~20 FPS. On Pi 5 it can run every frame.
- Drowsiness always takes priority over emotion when setting the light profile — a sleeping user overrides a Happy emotion reading.
- The `Disgusted` and `Fear` classes from the original 7-class FER dataset are merged into the 5-class model — they map to `Angry` and `Sad` respectively at inference time.
- Press `R` anytime to recalibrate EAR if you change seats, lighting, or put on glasses.
