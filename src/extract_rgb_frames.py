import os
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# Parameters
input_dir = "raw_videos"   # Folder containing your 'real' and 'fake' folders
output_dir = "dataset"     # Folder to save extracted frames
frame_size = (224, 224)
frames_per_video = 32      # Fixed number of frames for uniformity

# Ensure output structure
os.makedirs(output_dir, exist_ok=True)

def extract_frames_from_video(video_path, save_dir, num_frames=frames_per_video):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices to sample
    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()

    saved = 0
    for idx in range(total_frames):
        success, frame = cap.read()
        if not success:
            break
        if idx == indices[saved]:
            frame = cv2.resize(frame, frame_size)
            save_path = save_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(save_path), frame)
            saved += 1
        if saved >= num_frames:
            break
    cap.release()

def process_videos(input_dir, output_dir):
    for label in ["real", "fake"]:
        class_input_path = Path(input_dir) / label
        class_output_path = Path(output_dir) / label
        class_output_path.mkdir(parents=True, exist_ok=True)

        for video_file in tqdm(os.listdir(class_input_path), desc=f"Processing {label}"):
            video_path = class_input_path / video_file
            video_stem = video_file.split('.')[0]
            save_folder = class_output_path / video_stem
            save_folder.mkdir(exist_ok=True)
            extract_frames_from_video(video_path, save_folder)

if __name__ == "__main__":
    process_videos(input_dir, output_dir)
