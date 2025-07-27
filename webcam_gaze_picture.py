import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

MODEL_PATH  = "mobilenet_model.keras"
TARGET_SIZE = 224

# 1) Load model (if you also want to run inference on the still image)
model = tf.keras.models.load_model(MODEL_PATH)

# 2) MediaPipe FaceMesh setup
mpfm      = mp.solutions.face_mesh
face_mesh = mpfm.FaceMesh(
    static_image_mode=True,      # <-- single image mode
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

LEFT_EYE   = [33,133]; RIGHT_EYE  = [362,263]
UPPER_LIDS = [159,386]; LOWER_LIDS = [145,374]

def pad_and_resize(crop, size=TARGET_SIZE):
    h, w = crop.shape[:2]
    s = max(h, w)
    t, b = (s-h)//2, s-h-(s-h)//2
    l, r = (s-w)//2, s-w-(s-w)//2
    sq = cv2.copyMakeBorder(crop, t, b, l, r, cv2.BORDER_CONSTANT, value=[0,0,0])
    return cv2.resize(sq, (size, size))

def crop_eyes(frame):
    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark

    xs   = [int(lm[i].x * w) for i in LEFT_EYE+RIGHT_EYE]
    yts  = [int(lm[i].y * h) for i in UPPER_LIDS]
    ybs  = [int(lm[i].y * h) for i in LOWER_LIDS]

    x1, x2 = max(0, min(xs) - 25), min(w, max(xs) + 25)
    y1, y2 = max(0, min(yts) - 25), min(h, max(ybs) + 25)

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return pad_and_resize(patch)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    # Capture one frame
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Failed to capture image.")
        return

    # Save & show the original
    cv2.imwrite("capture_full.jpg", frame)
    cv2.imshow("Full Frame Capture", frame)

    # Crop eyes
    eye_patch = crop_eyes(frame)
    if eye_patch is not None:
        cv2.imwrite("capture_eyes.jpg", eye_patch)
        cv2.imshow("Cropped Eyes", eye_patch)
    else:
        print("⚠️ No face/eyes detected.")

    # Wait for any key, then clean up
    print("Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
