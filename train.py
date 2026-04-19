"""
train.py — Boucle d'entraînement complète avec évaluation et sauvegarde
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import load_ravdess, EMOTIONS, N_CLASSES
from model import EmotionCNN, count_parameters


# ─── Configuration ────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="Train EmotionCNN on RAVDESS")
    parser.add_argument("--data_dir",   type=str,   default="./data/RAVDESS")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--save_dir",   type=str,   default="./checkpoints")
    parser.add_argument("--feature",    type=str,   default="mel",
                        choices=["mel", "mfcc"])
    return parser.parse_args()


# ─── Une époque d'entraînement ────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


# ─── Évaluation (val ou test) ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train loss")
    ax1.plot(val_losses,   label="Val loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")

    ax2.plot([a * 100 for a in train_accs], label="Train acc")
    ax2.plot([a * 100 for a in val_accs],   label="Val acc")
    ax2.set_title("Accuracy (%)"); ax2.legend(); ax2.set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"Courbes sauvegardées → {path}")
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, save_dir):
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Matrice de confusion (normalisée)")

    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Matrice de confusion → {path}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # 1. Données
    train_loader, val_loader, test_loader = load_ravdess(
        args.data_dir, feature_type=args.feature
    )

    # 2. Modèle
    model = EmotionCNN(n_classes=N_CLASSES, dropout=args.dropout).to(device)
    print(f"Paramètres : {count_parameters(model):,}")

    # 3. Optimiseur + perte
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5,
                                  factor=0.5)

    # 4. Boucle d'entraînement
    best_val_loss = float("inf")
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)
        scheduler.step(vl_loss)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss: {tr_loss:.4f} acc: {tr_acc*100:.1f}% | "
              f"Val loss: {vl_loss:.4f} acc: {vl_acc*100:.1f}%")

        # Sauvegarde du meilleur modèle
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            ckpt = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl_loss,
                "val_acc": vl_acc,
                "feature_type": args.feature,
            }, ckpt)
            print(f"  ✓ Meilleur modèle sauvegardé (val_loss={vl_loss:.4f})")

    # 5. Évaluation finale sur le test set
    print("\n── Évaluation finale (test set) ──")
    ckpt = torch.load(os.path.join(args.save_dir, "best_model.pth"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, all_preds, all_labels = evaluate(model, test_loader,
                                                   criterion, device)
    print(f"Test accuracy : {test_acc*100:.2f}%\n")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS))

    # 6. Plots
    plot_history(train_losses, val_losses, train_accs, val_accs, args.save_dir)
    plot_confusion_matrix(all_labels, all_preds, args.save_dir)


if __name__ == "__main__":
    main()
