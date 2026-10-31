#  Object Tracking & Unique Counting

## Overview
This project implements **YOLOv8** combined with **ByteTrack** to assign stable tracking IDs to individual pedestrians across video frames and calculate the total unique person count.

---

## Repository Contents

* `track_and_count.py`: Main Python script running YOLOv8 + ByteTrack object tracking.
* `REPORT.md`: Detailed assignment write-up covering counting discrepancies and ID switch analysis.
* `Pedestrians_crossing_outdoor_plaza_7460bf97.mp4`: Input test video.
* `output_tracked.mp4`: Output video with bounding boxes, persistent track IDs, and real-time total unique counts.

---

## Technical Setup & Usage

### 1. Requirements
```bash
pip install ultralytics opencv-python
