import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from scipy.spatial import distance as dist
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    num_faces=1,
    output_face_blendshapes=True
)
detector    = vision.FaceLandmarker.create_from_options(options)
emotion_model = load_model('./face_model.h5')

class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


LEFT_EYE    = [33, 160, 158, 133, 153, 144]
RIGHT_EYE   = [362, 385, 387, 263, 373, 380]
NOSE_TIP    = 1
CHIN        = 152
LEFT_EYE_C  = 33
RIGHT_EYE_C = 263

#Eye Aspect Ration
def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5]) #A,B vertical distances and c horizontal distance
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def eyes_genuinely_closed(left_ear, right_ear, threshold, max_asymmetry=0.07):
    return (left_ear < threshold and right_ear < threshold and
            abs(left_ear - right_ear) < max_asymmetry)

def estimate_head_pitch(landmarks, w, h):
    l_eye      = np.array([landmarks[LEFT_EYE_C].x * w,  landmarks[LEFT_EYE_C].y * h])
    r_eye      = np.array([landmarks[RIGHT_EYE_C].x * w, landmarks[RIGHT_EYE_C].y * h])
    chin       = np.array([landmarks[CHIN].x * w,         landmarks[CHIN].y * h])
    eye_center = (l_eye + r_eye) / 2
    vertical   = chin - eye_center
    return np.degrees(np.arctan2(vertical[0], vertical[1]))

def get_light_profile(sleep_level, emotion):


    if sleep_level == "SLEEPING":
        return dict(brightness=10, temperature='warm',
                    color_hint=(0, 60, 80),
                    reason="Sleeping — minimal warm light")

    if sleep_level == "DROWSY":
        return dict(brightness=30, temperature='warm',
                    color_hint=(0, 100, 140),
                    reason="Drowsy — soft warm light to avoid strain")

    # Awake, decide by emotion
    profiles = {
        'Angry':     dict(brightness=45, temperature='warm',
                        color_hint=(0, 80, 180),
                        reason="Angry — dim warm to calm down"),
        'Disgusted': dict(brightness=50, temperature='warm',
                        color_hint=(0, 90, 170),
                        reason="Disgusted — soft warm light"),
        'Fear':      dict(brightness=60, temperature='warm',
                        color_hint=(0, 120, 200),
                        reason="Fear — warm reassuring light"),
        'Sad':       dict(brightness=55, temperature='warm',
                        color_hint=(0, 100, 190),
                        reason="Sad — gentle warm light"),
        'Surprise':  dict(brightness=90, temperature='cool',
                        color_hint=(200, 180, 0),
                        reason="Surprise — bright alert light"),
        'Happy':     dict(brightness=80, temperature='neutral',
                        color_hint=(0, 200, 200),
                        reason="Happy — natural balanced light"),
        'Neutral':   dict(brightness=75, temperature='cool',
                        color_hint=(180, 200, 0),
                        reason="Neutral/focused — cool productive light"),
    }
    return profiles.get(emotion, profiles['Neutral'])


def calibrate(cap, detector):
    print("Calibrating — keep eyes open for 3 seconds...")
    samples = []
    while len(samples) < 90:
        ret, frame = cap.read()
        if not ret:
            continue
        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _  = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        if result.face_landmarks:
            lm        = result.face_landmarks[0]
            left_eye  = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE]
            ear       = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0
            samples.append(ear)

        progress = int((len(samples) / 90) * 300)
        cv2.rectangle(frame, (30, 80), (330, 110), (40, 40, 40), -1)
        cv2.rectangle(frame, (30, 80), (30 + progress, 110), (0, 255, 255), -1)
        cv2.putText(frame, "CALIBRATING — eyes open", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("System", frame)
        cv2.waitKey(1)

    # baseline = np.mean(samples)
    # use median instead of mean in case person blink while calibrating
    baseline=np.median(samples)
    thresh   = baseline * 0.60
    print(f"Baseline EAR: {baseline:.4f}  →  Threshold: {thresh:.4f}")
    return thresh

BLINK_FRAMES    = 6
DROWSY_FRAMES   = 45
SLEEP_FRAMES    = 90
PERCLOS_WINDOW  = 90
PERCLOS_THRESH  = 0.35
PITCH_THRESH    = 20
NOD_FRAMES      = 20
RECOVERY_FRAMES = 8
MAX_ASYMMETRY   = 0.07
EMOTION_SMOOTH  = 10   

ear_history      = deque(maxlen=PERCLOS_WINDOW)
emotion_history  = deque(maxlen=EMOTION_SMOOTH)
closed_counter   = 0
open_counter     = 0
nod_counter      = 0
drowsy_score     = 0.0
sleep_level      = "AWAKE"
current_emotion  = "Neutral"

current_brightness = 75.0
target_brightness  = 75.0
SMOOTH_RATE        = 0.08  
cap           = cv2.VideoCapture(0)
EAR_THRESHOLD = calibrate(cap, detector)
ptime         = 0
frame_count   = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame      = cv2.flip(frame, 1)
    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w, _    = frame.shape
    frame_count += 1

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    face_detected = False

    if result.face_landmarks:
        face_detected = True
        landmarks     = result.face_landmarks[0]

        left_eye  = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
        left_ear  = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        ear       = (left_ear + right_ear) / 2.0

        eyes_closed = eyes_genuinely_closed(left_ear, right_ear,
                                            EAR_THRESHOLD, MAX_ASYMMETRY)

        ear_history.append(1 if eyes_closed else 0)
        perclos = sum(ear_history) / len(ear_history)

        pitch   = estimate_head_pitch(landmarks, w, h)
        nodding = pitch > PITCH_THRESH

        if eyes_closed:
            closed_counter += 1
            open_counter    = 0
        else:
            open_counter   += 1
            if open_counter >= RECOVERY_FRAMES:
                closed_counter = 0

        nod_counter = nod_counter + 1 if nodding else max(0, nod_counter - 1)

        ear_score     = min(closed_counter / SLEEP_FRAMES, 1.0)
        perclos_score = min(perclos / PERCLOS_THRESH, 1.0)
        nod_score     = min(nod_counter / NOD_FRAMES, 1.0)
        drowsy_score  = min(0.5 * ear_score + 0.35 * perclos_score + 0.15 * nod_score, 1.0)

        if closed_counter >= SLEEP_FRAMES or drowsy_score > 0.92:
            sleep_level = "SLEEPING"
        elif (closed_counter >= DROWSY_FRAMES and closed_counter > BLINK_FRAMES) \
                or perclos > PERCLOS_THRESH \
                or nod_counter >= NOD_FRAMES \
                or drowsy_score > 0.75:
            sleep_level = "DROWSY"
        else:
            sleep_level = "AWAKE"

        # Draw eye landmarks
        for pt in left_eye + right_eye:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

    if face_detected and frame_count % 3 == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                            minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, fw, fh   = faces[0]
            face_roi        = frame[y:y + fh, x:x + fw]
            face_img        = cv2.resize(face_roi, (48, 48))
            face_img        = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img        = keras_image.img_to_array(face_img)
            face_img        = np.expand_dims(face_img, axis=0)
            predictions     = emotion_model.predict(face_img, verbose=0)
            detected_emotion = class_names[np.argmax(predictions)]
            emotion_history.append(detected_emotion)

        if emotion_history:
            current_emotion = max(set(emotion_history), key=emotion_history.count)

    profile           = get_light_profile(sleep_level, current_emotion)
    target_brightness = profile['brightness']

    current_brightness += (target_brightness - current_brightness) * SMOOTH_RATE
    current_brightness  = np.clip(current_brightness, 0, 100)

    overlay     = frame.copy()
    tint_color  = profile['color_hint']
    tint_layer  = np.full_like(frame, tint_color, dtype=np.uint8)
    alpha       = 0.15  # subtle tint
    frame       = cv2.addWeighted(frame, 1 - alpha, tint_layer, alpha, 0)

    brightness_factor = current_brightness / 75.0   # 75 = neutral baseline
    frame = np.clip(frame.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    sleep_colors = {"AWAKE": (0,255,0), "DROWSY": (0,165,255),
                    "SLEEPING": (0,0,255), "NO FACE": (128,128,128)}
    sleep_color  = sleep_colors.get(sleep_level, (255,255,255))

    cv2.putText(frame, sleep_level, (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, sleep_color, 3)

    if face_detected:
        cv2.putText(frame, f"EAR: {ear:.3f}  thr:{EAR_THRESHOLD:.3f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
        cv2.putText(frame, f"L:{left_ear:.3f} R:{right_ear:.3f} diff:{abs(left_ear-right_ear):.3f}",
                    (30, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,220,0), 1)
        cv2.putText(frame, f"PERCLOS:{perclos:.0%}  closed:{closed_counter}f",
                    (30, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,200,0), 1)
        cv2.putText(frame, f"Pitch:{pitch:.1f}  {'NOD' if nodding else ''}",
                    (30, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,255), 1)


        bar_w     = int(200 * drowsy_score)
        bar_color = (0,255,0) if drowsy_score < 0.4 else \
                    (0,165,255) if drowsy_score < 0.75 else (0,0,255)
        cv2.rectangle(frame, (30, 165), (230, 180), (50,50,50), -1)
        cv2.rectangle(frame, (30, 165), (30 + bar_w, 180), bar_color, -1)
        cv2.putText(frame, f"Drowsy:{drowsy_score:.2f}",
                    (30, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    emotion_colors = {
        'Angry':'(0,0,220)', 'Disgusted':'(0,140,0)', 'Fear':'(130,0,130)',
        'Happy':'(0,200,200)', 'Sad':'(200,100,0)', 'Surprise':'(0,200,255)',
        'Neutral':'(180,180,180)'
    }
    emo_color_map = {
        'Angry':(0,0,220), 'Disgusted':(0,140,0), 'Fear':(130,0,130),
        'Happy':(0,200,200), 'Sad':(200,100,0), 'Surprise':(0,200,255),
        'Neutral':(180,180,180)
    }
    cv2.putText(frame, f"Emotion: {current_emotion}", (30, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                emo_color_map.get(current_emotion, (255,255,255)), 2)

    panel_y = h - 90
    cv2.rectangle(frame, (0, panel_y), (w, h), (20, 20, 20), -1)

    temp_color = (100,180,255) if profile['temperature'] == 'warm' else \
                 (255,255,200) if profile['temperature'] == 'cool' else (200,220,200)

    cv2.putText(frame, f"LIGHT: {int(current_brightness)}%  [{profile['temperature'].upper()}]",
                (20, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, temp_color, 2)
    cv2.putText(frame, profile['reason'],
                (20, panel_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    bbar_w = int((w - 40) * current_brightness / 100)
    cv2.rectangle(frame, (20, panel_y + 68), (w - 20, panel_y + 80), (50,50,50), -1)
    cv2.rectangle(frame, (20, panel_y + 68), (20 + bbar_w, panel_y + 80), temp_color, -1)

    ctime = time.time()
    fps   = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
    ptime = ctime
    cv2.putText(frame, f"FPS:{int(fps)}", (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Smart Light System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r'):
        EAR_THRESHOLD  = calibrate(cap, detector)
        closed_counter = 0
        open_counter   = 0
        ear_history.clear()
        emotion_history.clear()

cap.release()
cv2.destroyAllWindows()
