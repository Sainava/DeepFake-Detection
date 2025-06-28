import os

RGB_DIR = "./dataset/rgb"
OF_DIR = "./dataset/optical_flow"

total_rgb = 0
total_matched = 0
videos_missing = []

for label_str in ["real", "fake"]:
    rgb_label = os.path.join(RGB_DIR, label_str)
    of_label = os.path.join(OF_DIR, label_str)

    for video in os.listdir(rgb_label):
        rgb_vid = os.path.join(rgb_label, video)
        of_vid = os.path.join(of_label, video)
        if not os.path.isdir(rgb_vid):
            continue

        rgb_frames = [f for f in os.listdir(rgb_vid) if f.endswith(".jpg")]
        matched = 0

        for img in rgb_frames:
            of_img = img.replace("frame_", "flow_")
            of_path = os.path.join(of_vid, of_img)
            if os.path.exists(of_path):
                matched += 1

        total_rgb += len(rgb_frames)
        total_matched += matched

        if matched < len(rgb_frames):
            videos_missing.append((video, len(rgb_frames), matched))

print(f"\n[INFO] Total RGB frames: {total_rgb}")
print(f"[INFO] Total with matching flow: {total_matched}")
print(f"[INFO] Overall match: {total_matched/total_rgb:.2%}")

print("\n[INFO] Videos with missing matches:")
for v in videos_missing:
    print(f"  Video: {v[0]} | RGB frames: {v[1]} | Matched: {v[2]}")
