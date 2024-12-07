import cv2
import os

video_path = "data/video/balonmano.mp4"
output_folder = "data/frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(output_folder, f"frame_{frame_count:06d}.PNG")
    cv2.imwrite(frame_path, frame)
    frame_count += 1

cap.release()
print(f"Se extrajeron {frame_count} frames en {output_folder}")