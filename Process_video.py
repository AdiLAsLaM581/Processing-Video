import cv2

# 1. Read the input video using cv2.VideoCapture[cite: 8]
input_video_path = "E:\\Zee Project\\test.mp4"
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
  print("Error: Could not open video.")
  exit()

# Get video properties (width, height, frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 2. Set up VideoWriter to save the output video[cite: 8]
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video_path, fourcc, fps, (frame_width, frame_height)
)

frame_count = 0

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  frame_count += 1

  # 3. Draw a box on each frame using cv2.rectangle[cite: 8]
  # Parameters: frame, start_point, end_point, color (BGR), thickness
  cv2.rectangle(frame, (50, 50), (350, 150), (0, 255, 0), 2)

  # 4. Write text on each frame using cv2.putText[cite: 8]
  # Parameters: frame, text, origin, font, scale, color (BGR), thickness
  text = f"Frame: {frame_count}"
  cv2.putText(
      frame,
      text,
      (60, 110),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 0, 255),
      2,
      cv2.LINE_AA,
  )

  # Write the processed frame to the output file
  out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete. Saved as {output_video_path}")