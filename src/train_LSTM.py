import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === CONFIGURATION ===
FEATURE_DIR = './features/mixed'
SAVE_DIR = './models/lstm_model'
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_lstm.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
HIDDEN_SIZE = 512
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
VAL_SPLIT = 0.2

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

# === DATASET ===
class VideoFeatureDataset(Dataset):
    def __init__(self, file_list, labels):
        self.files = file_list
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        features = np.load(path)  # (T, feature_dim)
        features = torch.from_numpy(features).float()
        label = self.labels[idx]
        return features, label

def load_feature_paths(feature_dir):
    paths, labels = [], []
    for fname in os.listdir(feature_dir):
        if not fname.endswith('.npy'):
            continue
        label = 0 if fname.startswith('real_') else 1
        paths.append(os.path.join(feature_dir, fname))
        labels.append(label)
    return paths, labels

# === MODEL ===
class VideoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * factor, num_classes)
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h = h_n[-1]
        return self.classifier(h)

# === TRAINING ===
if __name__ == "__main__":
    paths, labels = load_feature_paths(FEATURE_DIR)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.1, stratify=labels, random_state=SEED
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=VAL_SPLIT, stratify=train_labels, random_state=SEED
    )

    datasets = {
        'train': VideoFeatureDataset(train_paths, train_labels),
        'val':   VideoFeatureDataset(val_paths, val_labels),
        'test':  VideoFeatureDataset(test_paths, test_labels)
    }
    loaders = {
        phase: DataLoader(datasets[phase], batch_size=BATCH_SIZE, shuffle=(phase == 'train'))
        for phase in ['train', 'val', 'test']
    }

    sample_feat = np.load(train_paths[0])
    FEATURE_DIM = sample_feat.shape[1]

    model = VideoLSTM(
        input_size=FEATURE_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_correct = 0
            total = 0
            for feats, labels in tqdm(loaders[phase], desc=phase):
                feats = feats.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(feats)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * feats.size(0)
                running_correct += (preds == labels).sum().item()
                total += feats.size(0)
            epoch_loss = running_loss / total
            epoch_acc = running_correct / total
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
            if phase == 'val':
                scheduler.step(epoch_acc)
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print(f"[INFO] Saved best model (val_acc = {best_val_acc:.4f})")

    print("\n=== Testing on held-out set ===")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    test_loss = 0.0
    test_correct = 0
    total = 0
    with torch.no_grad():
        for feats, labels in tqdm(loaders['test'], desc='test'):
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(feats)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            test_loss += loss.item() * feats.size(0)
            test_correct += (preds == labels).sum().item()
            total += feats.size(0)
    print(f"Test Loss: {test_loss/total:.4f} | Acc: {test_correct/total:.4f}")
