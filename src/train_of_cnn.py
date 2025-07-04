import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# ==== CONFIG ====
DATA_DIR = os.path.join("..", "dataset", "optical_flow")
SAVE_DIR = os.path.join("..", "models", "of_cnn")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
IMG_SIZE = 224
BATCH_SIZE = 16  # Smaller batch
EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
NUM_WORKERS = 2
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ==== TRANSFORMS ====
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 64, IMG_SIZE + 64)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ==== DATASET ====
class OpticalFlowDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_video_grouped_samples(data_dir):
    grouped = defaultdict(list)
    for label_str, label in [('real', 0), ('fake', 1)]:
        class_dir = os.path.join(data_dir, label_str)
        for video in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video)
            if not os.path.isdir(video_path): continue
            for img in os.listdir(video_path):
                if img.endswith(".jpg"):
                    grouped[f"{label_str}/{video}"].append((os.path.join(video_path, img), label))
    return grouped

def stratified_split(grouped, val_split, test_split):
    keys = list(grouped.keys())
    random.shuffle(keys)
    total = len(keys)
    test_k = int(test_split * total)
    val_k = int(val_split * total)

    test_keys = keys[:test_k]
    val_keys = keys[test_k:test_k+val_k]
    train_keys = keys[test_k+val_k:]

    train = sum([grouped[k] for k in train_keys], [])
    val = sum([grouped[k] for k in val_keys], [])
    test = sum([grouped[k] for k in test_keys], [])
    return train, val, test

# ==== MODEL ====
class ResNet50Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.base.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.base.fc.in_features, 2)
        )

    def forward(self, x):
        return self.base(x)

# ==== TRAINING ====
def train(model, train_dl, val_dl, optimizer, criterion, scheduler):
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct = 0.0, 0
        for imgs, labels in tqdm(train_dl, desc=f"[Epoch {epoch+1}/{EPOCHS}] Training"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_dl.dataset)
        print(f"[TRAIN] Loss: {total_loss:.4f} | Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_dl.dataset)
        val_loss /= len(val_dl)
        print(f"[VAL] Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        scheduler.step()

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[INFO] 🧠 Best model saved at epoch {epoch+1}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"[INFO] ⏹ Early stopping at epoch {epoch+1}")
                break

# ==== EVALUATION ====
def evaluate(model, test_dl):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
    acc = correct / len(test_dl.dataset)
    print(f"[TEST] Acc: {acc:.4f}")

# ==== MAIN ====
if __name__ == "__main__":
    print("[INFO] Preparing dataset...")
    grouped = get_video_grouped_samples(DATA_DIR)
    train_samples, val_samples, test_samples = stratified_split(grouped, VAL_SPLIT, TEST_SPLIT)

    train_ds = OpticalFlowDataset(train_samples, transform=train_transform)
    val_ds   = OpticalFlowDataset(val_samples, transform=eval_transform)
    test_ds  = OpticalFlowDataset(test_samples, transform=eval_transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("[INFO] Building model...")
    model = ResNet50Dropout().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("[INFO] Starting training...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    train(model, train_dl, val_dl, optimizer, criterion, scheduler)

    print("[INFO] Loading best model and evaluating on test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    evaluate(model, test_dl)
