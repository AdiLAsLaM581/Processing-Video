import os
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Input/Output paths using raw strings
video_path = r"E:\Zee Project\Track_And_Count\Pedestrians_crossing_outdoor_plaza_7460bf97.mp4"
output_path = r"E:\Zee Project\Track_And_Count\output_tracked.mp4"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

seen_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.25,  # Lower confidence threshold slightly if detections are missed
    )

    # Check if boxes and IDs are present
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            seen_ids.add(track_id)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Annotate total count
    cv2.putText(
        frame,
        f"Unique Count: {len(seen_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total unique IDs tracked: {len(seen_ids)}")
