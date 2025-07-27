import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

MODEL_PATH  = "model_xtra_drop_lay_+_40thresh.keras"
TARGET_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

# 2) MediaPipe FaceMesh (static_image_mode since we grab only one frame)
mpfm      = mp.solutions.face_mesh
face_mesh = mpfm.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# landmark indices
LEFT_EYE   = [33, 133]
RIGHT_EYE  = [362,263]
UPPER_LIDS = [159,386]
LOWER_LIDS = [145,374]

def pad_and_resize(crop, size=TARGET_SIZE):
    h, w = crop.shape[:2]
    s = max(h, w)
    top    = (s - h) // 2
    bottom = s - h - top
    left   = (s - w) // 2
    right  = s - w - left
    square = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=[0,0,0])
    return cv2.resize(square, (size, size))

def crop_eyes(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    # --- 1) pull out all relevant pixel coords ---
    ids_all = LEFT_EYE + RIGHT_EYE + UPPER_LIDS + LOWER_LIDS
    coords = {
        i: np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)
        for i in ids_all
    }

    # --- 2) find eye centers & angle ---
    left_center  = (coords[LEFT_EYE[0]]  + coords[LEFT_EYE[1]])  / 2
    right_center = (coords[RIGHT_EYE[0]] + coords[RIGHT_EYE[1]]) / 2
    dx, dy = right_center - left_center
    angle  = np.degrees(np.arctan2(dy, dx))

    # --- 3) rotate full frame so eyes are level ---
    M       = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    aligned = cv2.warpAffine(frame, M, (w, h))

    # helper to transform a point by M
    def transform(pt):
        x, y = pt
        x_new = M[0,0]*x + M[0,1]*y + M[0,2]
        y_new = M[1,0]*x + M[1,1]*y + M[1,2]
        return np.array([x_new, y_new], dtype=np.float32)

    # --- 4) rotate all landmark coords ---
    rot_coords = {i: transform(pt) for i, pt in coords.items()}

    # --- 5) build tight bbox around lids in aligned image ---
    xs  = [rot_coords[i][0] for i in LEFT_EYE + RIGHT_EYE]
    yts = [rot_coords[i][1] for i in UPPER_LIDS]
    ybs = [rot_coords[i][1] for i in LOWER_LIDS]

    pad = 5
    x1 = max(int(min(xs)) - pad, 0)
    x2 = min(int(max(xs)) + pad, w)
    y1 = max(int(min(yts)) - pad, 0)
    y2 = min(int(max(ybs)) + pad, h)

    patch = aligned[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    return pad_and_resize(patch)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        eye = crop_eyes(frame)
        if eye is not None:
            inp  = eye.astype(np.float32)/255.0
            prob = float(model.predict(inp[None,...])[0,0])
            pred = "Looking at camera!" if prob>=0.4 else "LOOKING AWAY!!"
            color= (0,255,0) if prob>=0.4 else (0,0,255)
            txt   = f"{pred} ({prob:.2f})"
        else:
            txt, color = "No face", (0,0,255)

        cv2.putText(frame, txt, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Eye Gaze", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()