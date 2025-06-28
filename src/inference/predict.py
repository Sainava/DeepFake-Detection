import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# ======== CONFIG ========
RGB_DIR = "./predict_input/rgb"     # New video’s RGB frames here
OF_DIR = "./predict_input/optical_flow"  # New video’s Optical Flow frames here
MIXED_MODEL_PATH = "./models/mixed_model/best_mixed_model.pth"
LSTM_MODEL_PATH = "./models/lstm_model/best_lstm.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== TRANSFORMS ========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ======== ResNet Feature Extractor ========
class MixedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.rgb.fc = nn.Identity()
        self.of = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.of.fc = nn.Identity()

    def forward(self, rgb, of):
        rgb_feat = self.rgb(rgb)
        of_feat = self.of(of)
        fused = torch.cat((rgb_feat, of_feat), dim=1)
        return fused

# ======== LSTM Classifier ========
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        out = self.fc(output[:, -1, :])
        return out

# ======== Predict Function ========
def predict():
    # === Load models ===
    feature_extractor = MixedResNet().to(DEVICE)
    feature_extractor.load_state_dict(torch.load(MIXED_MODEL_PATH, map_location=DEVICE))
    feature_extractor.eval()

    lstm_model = LSTMClassifier().to(DEVICE)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
    lstm_model.eval()

    # === Gather frames ===
    rgb_frames = sorted(os.listdir(RGB_DIR))
    of_frames = sorted(os.listdir(OF_DIR))
    assert len(rgb_frames) == len(of_frames), "Mismatch in RGB & Optical Flow frames!"

    all_feats = []

    # === Extract features for each frame ===
    for rgb_img_name, of_img_name in tqdm(zip(rgb_frames, of_frames), total=len(rgb_frames), desc="Extracting"):
        rgb_path = os.path.join(RGB_DIR, rgb_img_name)
        of_path = os.path.join(OF_DIR, of_img_name)

        rgb = Image.open(rgb_path).convert("RGB")
        of = Image.open(of_path).convert("RGB")

        rgb = transform(rgb).unsqueeze(0).to(DEVICE)
        of = transform(of).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = feature_extractor(rgb, of)  # (1, 1024)
            all_feats.append(feat.cpu().numpy())

    # === Build (T, 1024) tensor ===
    features = np.vstack(all_feats)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, 1024)

    # === Predict ===
    with torch.no_grad():
        output = lstm_model(features)
        pred = torch.argmax(output, dim=1).item()

    label = "REAL" if pred == 0 else "FAKE"
    print(f"\n✅ Prediction: {label}")

if __name__ == "__main__":
    predict()
