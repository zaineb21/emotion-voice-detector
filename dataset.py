"""
dataset.py — Chargement RAVDESS + extraction features (MFCCs / Mel spectrogramme)
"""

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Labels
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
EMOTIONS = list(EMOTION_MAP.values())
N_CLASSES = len(EMOTIONS)


# Augmentation audio
def augment_audio(y, sr):
    if np.random.rand() < 0.5:
        y = y + np.random.randn(len(y)) * 0.005
    if np.random.rand() < 0.5:
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
    if np.random.rand() < 0.5:
        steps = np.random.randint(-2, 3)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    if np.random.rand() < 0.3:
        shift = np.random.randint(sr // 10)
        y = np.roll(y, shift)
    return y


def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, max_len=128, augment=False):
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    if augment:
        y = augment_audio(y, sr)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_len - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :max_len]

    return mel_db[np.newaxis, :, :]


def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=128):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc[np.newaxis, :, :]


class RAVDESSDataset(Dataset):
    def __init__(self, file_paths, labels, feature_type="mel", augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_type = feature_type
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        if self.feature_type == "mel":
            feat = extract_mel_spectrogram(path, augment=self.augment)
        else:
            feat = extract_mfcc(path)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_ravdess(data_dir, feature_type="mel"):
    file_paths, labels = [], []

    for actor_folder in sorted(os.listdir(data_dir)):
        actor_path = os.path.join(data_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            emotion_code = parts[2]
            if emotion_code not in EMOTION_MAP:
                continue
            labels.append(EMOTIONS.index(EMOTION_MAP[emotion_code]))
            file_paths.append(os.path.join(actor_path, fname))

    print(f"Total fichiers : {len(file_paths)}")
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo:12s} : {labels.count(i)}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        file_paths, labels, test_size=0.30, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    # Augmentation activée uniquement sur le train set
    train_ds = RAVDESSDataset(X_train, y_train, feature_type, augment=True)
    val_ds   = RAVDESSDataset(X_val,   y_val,   feature_type, augment=False)
    test_ds  = RAVDESSDataset(X_test,  y_test,  feature_type, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

    print(f"\nSplit -> train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    return train_loader, val_loader, test_loader