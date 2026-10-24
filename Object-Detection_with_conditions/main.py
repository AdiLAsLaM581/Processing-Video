import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load models
model_n = YOLO("yolo11n.pt")
model_x = YOLO("yolo11x.pt")

conf_thresholds = [0.2, 0.4, 0.6, 0.8]
input_video = "E:\Zee Project\Object-Detection_with_conditions\WALKING DANCE.mp4"

def get_counts(model, frame, conf):
    results = model(frame, conf=conf, classes=[0], verbose=False)[0]
    return len(results.boxes)

cap = cv2.VideoCapture(input_video)
ret, frame = cap.read()
cap.release()

if ret:
    # 1. Apply image transformations
    dark_frame = cv2.convertScaleAbs(frame, alpha=0.4, beta=0) # Darken
    blur_frame = cv2.GaussianBlur(frame, (21, 21), 0)         # Blur

    # 2. Sweep confidence across models and video conditions
    counts_n_normal = [get_counts(model_n, frame, c) for c in conf_thresholds]
    counts_x_normal = [get_counts(model_x, frame, c) for c in conf_thresholds]
    
    counts_n_dark = [get_counts(model_n, dark_frame, c) for c in conf_thresholds]
    counts_n_blur = [get_counts(model_n, blur_frame, c) for c in conf_thresholds]

    # Chart 1: Compare yolo11n vs yolo11x
    plt.figure(figsize=(8, 4))
    plt.plot(conf_thresholds, counts_n_normal, marker='o', label='yolo11n')
    plt.plot(conf_thresholds, counts_x_normal, marker='s', label='yolo11x')
    plt.title("Model Comparison (yolo11n vs yolo11x)")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Detections Count")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_comparison.png")
    plt.close()

    # Chart 2: Video condition impact (Normal vs Dark vs Blur)
    plt.figure(figsize=(8, 4))
    plt.plot(conf_thresholds, counts_n_normal, marker='o', label='Normal')
    plt.plot(conf_thresholds, counts_n_dark, marker='s', label='Darkened')
    plt.plot(conf_thresholds, counts_n_blur, marker='^', label='Blurred')
    plt.title("Video Condition Impact (yolo11n)")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Detections Count")
    plt.legend()
    plt.grid(True)
    plt.savefig("condition_impact.png")
    plt.close()

    print("Evaluation complete. Charts saved: model_comparison.png and condition_impact.png")