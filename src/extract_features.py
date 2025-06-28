import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
from train_mixed_model import MixedResNet, eval_transform, DEVICE, BEST_MODEL_PATH

# === CONFIG ===
RGB_DIR = "./dataset/rgb"
OF_DIR = "./dataset/optical_flow"
FEATURE_SAVE_DIR = "./features/mixed"
os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)

# === Load frozen model ===
model = MixedResNet().to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()  # no grad

# Remove final head, keep only feature extractor
model.head = torch.nn.Identity()

# === Extract for each video ===
with torch.no_grad():
    for label_str in ["real", "fake"]:
        rgb_label_dir = os.path.join(RGB_DIR, label_str)
        of_label_dir = os.path.join(OF_DIR, label_str)

        for video in os.listdir(rgb_label_dir):
            rgb_vid_dir = os.path.join(rgb_label_dir, video)
            of_vid_dir = os.path.join(of_label_dir, video)
            if not os.path.isdir(rgb_vid_dir):
                continue

            frames = sorted([f for f in os.listdir(rgb_vid_dir) if f.endswith(".jpg")])
            frames = frames[:-1]  # Drop last RGB frame

            video_feats = []
            for frame in tqdm(frames, desc=f"Video {video}"):
                rgb_path = os.path.join(rgb_vid_dir, frame)
                of_name = frame.replace("frame_", "flow_")
                of_path = os.path.join(of_vid_dir, of_name)

                rgb = Image.open(rgb_path).convert("RGB")
                of  = Image.open(of_path).convert("RGB")
                rgb = eval_transform(rgb).unsqueeze(0).to(DEVICE)
                of  = eval_transform(of).unsqueeze(0).to(DEVICE)

                fused_feat = model(rgb, of).squeeze().cpu().numpy()
                video_feats.append(fused_feat)

            feats_array = np.stack(video_feats)  # (T, 1024)
            save_name = f"{label_str}_{video}.npy"
            np.save(os.path.join(FEATURE_SAVE_DIR, save_name), feats_array)

print("Done! Features saved to:", FEATURE_SAVE_DIR)
