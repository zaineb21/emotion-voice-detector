"""
model.py — Architecture CNN 2D pour classification d'émotions vocales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    """
    CNN 2D appliqué sur un mel spectrogramme (1, 128, 128).

    Architecture :
        3 blocs Conv → BatchNorm → ReLU → MaxPool → Dropout
        Flatten → FC(256) → FC(128) → FC(n_classes)

    Pourquoi ce choix ?
    - Le spectrogramme mel est une image 2D : fréquences × temps.
    - Les Conv2d capturent des patterns locaux (consonnes, formants, intonations).
    - BatchNorm stabilise l'entraînement sur un dataset de taille modeste.
    - Dropout réduit l'overfitting (dataset ~1500 fichiers).
    """

    def __init__(self, n_classes: int = 8, dropout: float = 0.3):
        super(EmotionCNN, self).__init__()

        # ── Bloc 1 : (1, 128, 128) → (32, 64, 64) ──────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),   # same padding
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128→64
            nn.Dropout2d(p=dropout),
        )

        # ── Bloc 2 : (32, 64, 64) → (64, 32, 32) ───────────────────────────
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # 64→32
            nn.Dropout2d(p=dropout),
        )

        # ── Bloc 3 : (64, 32, 32) → (128, 16, 16) ──────────────────────────
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # 32→16
            nn.Dropout2d(p=dropout),
        )

        # ── Global Average Pooling → shape fixe quelle que soit l'entrée ────
        self.gap = nn.AdaptiveAvgPool2d((4, 4))     # → (128, 4, 4)

        # ── Tête de classification ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # 128*4*4 = 2048
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_classes),              # logits (pas de softmax ici)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 1, 128, 128)
        retourne : (batch, n_classes) — logits bruts
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités via softmax (pour la démo Streamlit)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


# ─── Fonction utilitaire : résumé du modèle ───────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EmotionCNN(n_classes=8)
    print(model)
    dummy = torch.randn(4, 1, 128, 128)   # batch de 4 exemples
    out = model(dummy)
    print(f"\nInput shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Paramètres   : {count_parameters(model):,}")
