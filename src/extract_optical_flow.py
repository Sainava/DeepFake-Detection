import os
import cv2
import numpy as np
from tqdm import tqdm
from utils.optical_flow_utils import compute_optical_flow, normalize_flow, save_flow_as_image

INPUT_DIR = os.path.join("..", "dataset")
OUTPUT_DIR = os.path.join(INPUT_DIR, 'optical_flow')

def get_sorted_frame_paths(video_folder):
    frames = [f for f in os.listdir(video_folder) if f.startswith("frame_") and f.endswith(".jpg")]
    return sorted(frames)

def process_video(video_path, output_path):
    frame_files = get_sorted_frame_paths(video_path)
    os.makedirs(output_path, exist_ok=True)

    for i in range(len(frame_files) - 1):
        frame1_path = os.path.join(video_path, frame_files[i])
        frame2_path = os.path.join(video_path, frame_files[i + 1])

        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)

        # Convert to grayscale for optical flow
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = compute_optical_flow(gray1, gray2)
        norm_flow = normalize_flow(flow)

        flow_save_path = os.path.join(output_path, f'flow_{i:04d}.jpg')
        save_flow_as_image(norm_flow, flow_save_path)

def main():
    for label in ['real', 'fake']:
        label_dir = os.path.join(INPUT_DIR, label)
        for video_name in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
            video_path = os.path.join(label_dir, video_name)
            output_path = os.path.join(OUTPUT_DIR, label, video_name)

            if os.path.isdir(video_path):
                process_video(video_path, output_path)

if __name__ == "__main__":
    main()
