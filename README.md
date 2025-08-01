# Computer-Vision-Determining-Where-Eyes-Look

# Gaze Detection

A computer vision project that detects whether a person is looking directly at the camera or not, using deep learning and real-time video input.

## Overview

This project implements **binary gaze classification** — identifying if a person is **looking at the camera** or **not**. It uses a CNN model refined with **MobileNet** and **Mediapipe eye detection** to improve accuracy and reduce background noise.

The final model was trained on the **Columbia Gaze Dataset**, with performance improvements achieved through:
- Face cropping with Mediapipe
- MobileNet as backbone
- Regularization techniques (dropout, L2, noise)
- Real-time webcam prediction

## Dataset

- **Source**: Columbia Gaze Dataset
- **Size**: 5,880 headshot photos
- **Class Imbalance**: Only ~720 images of subjects looking at the camera vs. ~5,000 in the "not looking" class

---

Medium Article: https://medium.com/@aryan16/determining-where-eyes-look-using-cnns-to-detect-gazes-04d4c56b18db


## Project Architecture

### 📂 Gaze Detection Repository Structure

- `webcam_gaze1.py` – Run real-time webcam gaze detection  
- `webcam_gaze_picture.py` – Run gaze detection on a single captured frame  
- `capture_full.jpg` – Original captured frame from webcam  
- `capture_eyes.jpg` – Cropped version of eyes (Mediapipe)  
- `capture_eyes_pred.jpg` – Image showing gaze prediction output  
- `Gaze Detection.pdf` – Project presentation (overview, methodology, results)


---

## How to Use

### 1. Install Requirements

```bash pip install opencv-python mediapipe tensorflow numpy ``` 



### 2. Run Webcam Gaze Detection (Live)
 
 ```bash python webcam_gaze1.py ```

This will:
- Launch the webcam
- Detect eyes using Mediapipe
- Classify gaze direction (camera vs. not)
- Display the result in real time

### 3. Run on a Single Frame
   
```bash python webcam_gaze_picture.py ```

This:
- Captures one frame from the webcam
- Crops eyes
- Predicts and visualizes gaze direction
