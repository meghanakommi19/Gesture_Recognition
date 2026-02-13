from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# ---------------- MediaPipe ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_draw = mp.solutions.drawing_utils

# --------------------------------------------------
# FEATURE FORMAT:
# [Thumb, Index, Middle, Ring, Pinky, Direction]
# Direction: -1 = Left/Down, 0 = Neutral, 1 = Right/Up
# --------------------------------------------------

X = np.array([
    [0,0,0,0,0, 0],   # Fist
    [1,1,1,1,1, 0],   # Open Palm

    [1,0,0,0,0, 1],   # Thumbs Up
    [1,0,0,0,0,-1],   # Thumbs Down

    [0,1,0,0,0, 1],   # Point Right
    [0,1,0,0,0,-1],   # Point Left

    [0,0,1,0,0, 0],   # Middle Finger
    [0,0,0,1,0, 0],   # Ring Finger
    [0,0,0,0,1, 0],   # Pinky

    [0,1,1,0,0, 0],   # Two Fingers
    [0,1,1,1,0, 0],   # Three Fingers
    [0,1,1,1,1, 0],   # Four Fingers

    [0,1,0,0,1, 0],   # Rock
    [1,0,0,0,1, 0],   # Call Me
    [1,1,0,0,0, 0],   # Love
    [1,0,1,0,1, 0],   # Spider-Man
    [1,1,1,1,1, 0],   # Stop
])

y = np.array([
    "Fist",
    "Open Palm",
    "Thumbs Up",
    "Thumbs Down",
    "Point Right",
    "Point Left",
    "Middle Finger",
    "Ring Finger",
    "Pinky",
    "Two Fingers",
    "Three Fingers",
    "Four Fingers",
    "Rock",
    "Call Me",
    "Love",
    "Spider-Man",
    "Stop"
])

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

TIP_IDS = [4, 8, 12, 16, 20]

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(lm, hand_label):
    features = []

    # Thumb open (left/right aware)
    if hand_label == "Right":
        thumb = 1 if lm[4].x < lm[3].x else 0
    else:
        thumb = 1 if lm[4].x > lm[3].x else 0
    features.append(thumb)

    # Other fingers
    for tip in TIP_IDS[1:]:
        features.append(1 if lm[tip].y < lm[tip-2].y else 0)

    # -------- Direction Feature --------
    direction = 0

    # Thumbs up / down
    if features[:5] == [1,0,0,0,0]:
        if lm[4].y < lm[2].y:
            direction = 1     # Up
        else:
            direction = -1    # Down

    # Point left / right
    if features[:5] == [0,1,0,0,0]:
        if lm[8].x > lm[5].x:
            direction = 1     # Right
        else:
            direction = -1    # Left

    features.append(direction)
    return features

# ---------------- CAMERA STREAM ----------------
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for i, hand in enumerate(result.multi_hand_landmarks):
                lm = hand.landmark
                label = result.multi_handedness[i].classification[0].label

                features = extract_features(lm, label)
                gesture = model.predict([features])[0]

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                cx = int(lm[0].x * w)
                cy = int(lm[0].y * h)

                cv2.putText(
                    frame,
                    f"{label}: {gesture}",
                    (cx - 40, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

# ---------------- FLASK ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
