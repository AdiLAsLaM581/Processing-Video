# Object Tracking & Counting Report

## 1. Unique Person Count Comparison
- **Manual Count:** 10 people
- **Automated Count (ByteTrack):** 31 people
- **Difference Analysis:** 
  The tracker overcounted by 21 IDs. This occurred because edge detections (people partially entering or exiting the frame borders) repeatedly broke and restarted tracks. Additionally, path crossings caused temporary occlusions where existing track IDs were lost and re-initialized as new numbers.

## 2. ID Switch Analysis
- **Timestamp:** 00:03 - 00:04
- **Observed Crossing:** Woman in yellow hoodie crossing paths with the man in the navy sweater/beige pants.
- **Cause:** 
  At 00:03, the man occludes the woman. ByteTrack relies purely on motion and bounding box IoU overlap without an appearance extraction model. The temporary loss of bounding box overlap during the crossing caused her track to drop and re-assign a new ID once she emerged at 00:04.