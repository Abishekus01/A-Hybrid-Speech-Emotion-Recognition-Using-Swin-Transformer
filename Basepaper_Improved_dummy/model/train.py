import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils.config import EMOTIONS, TESS_PATH, EMOVO_PATH, MODEL_WEIGHTS
from preprocessing.feature_extraction import extract_log_mel, extract_handcrafted
from model.swin_tser import SwinSERModel

# ======================
# CONFIG
# ======================
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
IMG_SIZE = (128, 128)
NUM_WORKERS = 0
DEVICE = torch.device("cpu")

# ======================
# FOCAL LOSS (BOOST ACC)
# ======================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, label_smoothing=smoothing)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss)

# ======================
# SPEC AUGMENT
# ======================
def spec_augment(mel):
    if torch.rand(1).item() < 0.5:
        time_mask = torch.randint(5, 20, (1,)).item()
        t0 = torch.randint(0, mel.shape[2] - time_mask, (1,)).item()
        mel[:, :, t0:t0 + time_mask] = 0

    if torch.rand(1).item() < 0.5:
        freq_mask = torch.randint(5, 20, (1,)).item()
        f0 = torch.randint(0, mel.shape[1] - freq_mask, (1,)).item()
        mel[:, f0:f0 + freq_mask, :] = 0

    return mel

# ======================
# DATASET
# ======================
class SERDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        log_mel = extract_log_mel(file_path, IMG_SIZE)
        handcrafted = extract_handcrafted(file_path)

        if handcrafted.shape[0] > 33:
            handcrafted = handcrafted[:33]
        elif handcrafted.shape[0] < 33:
            handcrafted = torch.cat([handcrafted, torch.zeros(33 - handcrafted.shape[0])])

        log_mel = log_mel.unsqueeze(0)

        if self.augment:
            log_mel = spec_augment(log_mel)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return log_mel, handcrafted, label

# ======================
# LOAD DATA
# ======================
def load_dataset():
    file_paths = []
    labels = []

    EMOVO_MAP = {
        "rab": "angry",
        "gio": "happy",
        "pau": "fear",
        "tri": "sad",
        "dis": "disgust",
        "sor": "surprise",
        "neu": "neutral"
    }

    def get_emotion(file, dataset_type):
        name = file.lower()
        if dataset_type == "TESS":
            return name.split("_")[-1].replace(".wav", "")
        elif dataset_type == "EMOVO":
            return EMOVO_MAP.get(name[:3], None)
        return None

    for root, _, files in os.walk(TESS_PATH):
        for file in files:
            if file.endswith(".wav"):
                emo = get_emotion(file, "TESS")
                if emo in EMOTIONS:
                    file_paths.append(os.path.join(root, file))
                    labels.append(EMOTIONS.index(emo))

    for root, _, files in os.walk(EMOVO_PATH):
        for file in files:
            if file.endswith(".wav"):
                emo = get_emotion(file, "EMOVO")
                if emo in EMOTIONS:
                    file_paths.append(os.path.join(root, file))
                    labels.append(EMOTIONS.index(emo))

    print("Total samples:", len(file_paths))
    return file_paths, labels

# ======================
# MIXUP
# ======================
def mixup(x1, x2, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x1.size(0)
    index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]

    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam

# ======================
# TRAIN
# ======================
def train():
    file_paths, labels = load_dataset()

    indices = list(range(len(file_paths)))
    np.random.shuffle(indices)

    split = int(0.85 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_dataset = SERDataset([file_paths[i] for i in train_idx],
                               [labels[i] for i in train_idx],
                               augment=True)

    val_dataset = SERDataset([file_paths[i] for i in val_idx],
                             [labels[i] for i in val_idx],
                             augment=False)

    # ===== WEIGHTED SAMPLER (BETTER THAN CLASS WEIGHT ONLY) =====
    train_labels = [labels[i] for i in train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_weights_tensor = torch.tensor(1.0 / np.bincount(labels), dtype=torch.float32).to(DEVICE)

    model = SwinSERModel(num_classes=len(EMOTIONS), handcrafted_dim=33).to(DEVICE)

    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2, smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for log_mel, handcrafted, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            log_mel = log_mel.to(DEVICE)
            handcrafted = handcrafted.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            if torch.rand(1).item() < 0.5:
                log_mel, handcrafted, y_a, y_b, lam = mixup(log_mel, handcrafted, labels_batch)
                outputs = model(log_mel, handcrafted)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(log_mel, handcrafted)
                loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # ===== VALIDATION WITH METRICS =====
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for log_mel, handcrafted, labels_batch in val_loader:
                log_mel = log_mel.to(DEVICE)
                handcrafted = handcrafted.to(DEVICE)

                outputs = model(log_mel, handcrafted)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels_batch.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        val_acc = (all_preds == all_labels).mean() * 100
        precision = precision_score(all_labels, all_preds, average="weighted") * 100
        recall = recall_score(all_labels, all_preds, average="weighted") * 100
        f1 = f1_score(all_labels, all_preds, average="weighted") * 100
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Acc: {val_acc:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%")
        print("Confusion Matrix:\n", cm)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_WEIGHTS)
            print("✅ Best model saved")

    print(f"\n➡️ Training Finished. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()