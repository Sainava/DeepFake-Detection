import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# === CONFIG ===
RGB_DIR = "./dataset/rgb"
OF_DIR = "./dataset/optical_flow"
SAVE_DIR = "./models/mixed_model"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_mixed_model.pth")
os.makedirs(SAVE_DIR, exist_ok=True)
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
NUM_WORKERS = 2
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 64, IMG_SIZE + 64)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

class MixedDataset(Dataset):
    def __init__(self, samples, transform_rgb, transform_of):
        self.samples = samples
        self.transform_rgb = transform_rgb
        self.transform_of = transform_of

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, of_path, label = self.samples[idx]
        rgb = Image.open(rgb_path).convert("RGB")
        of  = Image.open(of_path).convert("RGB")
        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        if self.transform_of:
            of = self.transform_of(of)
        return rgb, of, label

def get_mixed_samples(rgb_dir, of_dir):
    samples = []
    for label_str, label in [('real', 0), ('fake', 1)]:
        rgb_label = os.path.join(rgb_dir, label_str)
        of_label = os.path.join(of_dir, label_str)
        for video in os.listdir(rgb_label):
            rgb_vid = os.path.join(rgb_label, video)
            of_vid = os.path.join(of_label, video)
            if not os.path.isdir(rgb_vid) or not os.path.isdir(of_vid):
                continue
            frames = sorted([f for f in os.listdir(rgb_vid) if f.endswith(".jpg")])
            frames = frames[:-1]  # Drop last RGB frame since it has no flow
            for img in frames:
                rgb_path = os.path.join(rgb_vid, img)
                of_img_name = img.replace("frame_", "flow_")
                of_path = os.path.join(of_vid, of_img_name)
                samples.append((rgb_path, of_path, label))
    return samples



def stratified_split(samples, val_split, test_split):
    random.shuffle(samples)
    total = len(samples)
    test_n = int(test_split * total)
    val_n = int(val_split * total)
    test = samples[:test_n]
    val = samples[test_n:test_n+val_n]
    train = samples[test_n+val_n:]
    return train, val, test

class MixedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.rgb.fc = nn.Identity()
        self.of = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.of.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*2, 2)
        )

    def forward(self, rgb, of):
        rgb_feat = self.rgb(rgb)
        of_feat = self.of(of)
        fused = torch.cat((rgb_feat, of_feat), dim=1)
        return self.head(fused)

def train(model, train_dl, val_dl, optimizer, criterion, scheduler):
    best_val_acc, patience_counter = 0.0, 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct = 0.0, 0
        for rgb, of, labels in tqdm(train_dl, desc=f"[Epoch {epoch+1}/{EPOCHS}] Training"):
            rgb, of, labels = rgb.to(DEVICE), of.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(rgb, of)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_dl.dataset)
        print(f"[TRAIN] Loss: {total_loss:.4f} | Acc: {train_acc:.4f}")

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for rgb, of, labels in val_dl:
                rgb, of, labels = rgb.to(DEVICE), of.to(DEVICE), labels.to(DEVICE)
                outputs = model(rgb, of)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_dl.dataset)
        val_loss /= len(val_dl)
        print(f"[VAL] Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("[INFO] Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

def test(model, test_dl, criterion):
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    test_correct, test_loss = 0, 0.0
    with torch.no_grad():
        for rgb, of, labels in test_dl:
            rgb, of, labels = rgb.to(DEVICE), of.to(DEVICE), labels.to(DEVICE)
            outputs = model(rgb, of)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_correct += (outputs.argmax(1) == labels).sum().item()
    test_acc = test_correct / len(test_dl.dataset)
    test_loss /= len(test_dl)
    print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")


if __name__ == "__main__":
    samples = get_mixed_samples(RGB_DIR, OF_DIR)
    train_s, val_s, test_s = stratified_split(samples, VAL_SPLIT, TEST_SPLIT)

    train_ds = MixedDataset(train_s, train_transform, train_transform)
    val_ds   = MixedDataset(val_s, eval_transform, eval_transform)
    test_ds  = MixedDataset(test_s, eval_transform, eval_transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MixedResNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    train(model, train_dl, val_dl, optimizer, criterion, scheduler)
    test(model, test_dl, criterion)
