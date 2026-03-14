# 😄 Emotion Recognition

Real-time facial emotion detection using a custom CNN, trained on balanced FER dataset.
Detects 5 emotions: **Angry, Happy, Neutral, Sad, Surprise** — with 69% accuracy.

---

## 📋 Requirements

- Python 3.9+
- Webcam (for real-time inference)
- pip

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/lendadeif/Emotion-Recognition.git
cd emotion-recognition
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
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

Press `Q` to quit the webcam window.

---

## 📁 Project Structure

```
emotion-recognition/
├── main.py                 # Entry point — runs real-time detection
├── model/
│   └── FER_B.keras    # Trained model weights
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📦 requirements.txt

```
tensorflow>=2.12.0
opencv-python>=4.7.0
numpy>=1.23.0
```

---

## 🧠 Model

| Property       | Value                        |
|----------------|------------------------------|
| Architecture   | Custom CNN (from scratch)    |
| Input size     | 48×48 grayscale              |
| Classes        | 5 (angry, happy, neutral, sad, surprise) |
| Val Accuracy   | 69%                          |
| Macro F1       | 0.69                         |
| Format         | `.keras`                     |

---
