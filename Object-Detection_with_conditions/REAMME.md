# Week 3: Model Evaluation & Robustness Testing

This module evaluates object detection performance across different YOLO model sizes, confidence thresholds, and video degradation conditions.

## Key Objectives
- **Model Comparison:** Evaluates detection counts between `yolo11n` (nano) and `yolo11x` (extra-large).
- **Condition Testing:** Tests model performance under degraded video conditions (Normal vs. Darkened vs. Blurred).
- **Confidence Sweep:** Analyzes performance across confidence thresholds ($0.2, 0.4, 0.6, 0.8$)[cite: 16].

## Setup & Installation

```bash
pip install ultralytics opencv-python matplotlib
