# 🎙️ Emotion Voice Detector — RAVDESS + PyTorch CNN

**Projet Deep Learning — Aivancity MSc Data Engineering**
**Deadline : 20 avril 2026**

## Description

Détection d'émotions dans la voix à partir de fichiers audio.
Le modèle est un **CNN 2D PyTorch** entraîné sur des **mel spectrogrammes** extraits du dataset RAVDESS.

**8 émotions** : neutral, calm, happy, sad, angry, fearful, disgust, surprised

---

## Structure du projet

```
emotion_detection/
├── dataset.py          # Chargement RAVDESS + extraction mel/MFCC
├── model.py            # Architecture CNN 2D PyTorch
├── train.py            # Boucle d'entraînement + évaluation + plots
├── app.py              # Démo Streamlit interactive
├── requirements.txt
├── data/
│   └── RAVDESS/        # ← déposer ici le dataset téléchargé
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ...
└── checkpoints/        # Créé automatiquement à l'entraînement
    ├── best_model.pth
    ├── training_curves.png
    └── confusion_matrix.png
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset RAVDESS

1. Télécharger sur Kaggle : https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
2. Extraire dans `./data/RAVDESS/`
3. Structure attendue : `Actor_01/`, `Actor_02/`, ..., `Actor_24/`

**Format des noms de fichiers RAVDESS :**
`03-01-05-01-01-01-01.wav`
- Chiffre 3 : émotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)

---

## Entraînement

```bash
python train.py --data_dir ./data/RAVDESS --epochs 50 --lr 0.001
```

**Options :**
| Argument | Défaut | Description |
|---|---|---|
| `--data_dir` | `./data/RAVDESS` | Chemin du dataset |
| `--epochs` | `50` | Nombre d'époques |
| `--lr` | `0.001` | Learning rate |
| `--dropout` | `0.3` | Taux de dropout |
| `--feature` | `mel` | `mel` ou `mfcc` |
| `--save_dir` | `./checkpoints` | Dossier de sauvegarde |

---

## Démo Streamlit

```bash
streamlit run app.py
```

Ouvre http://localhost:8501 — upload un fichier `.wav` et vois l'émotion détectée en temps réel avec le spectrogramme et les probabilités.

---

## Architecture du modèle

```
Input: (batch, 1, 128, 128)  ← mel spectrogramme
  │
  ├─ Conv Block 1: Conv2d(1→32) + BN + ReLU × 2 + MaxPool + Dropout
  ├─ Conv Block 2: Conv2d(32→64) + BN + ReLU × 2 + MaxPool + Dropout
  ├─ Conv Block 3: Conv2d(64→128) + BN + ReLU × 2 + MaxPool + Dropout
  │
  ├─ Global Average Pooling → (128, 4, 4)
  │
  ├─ FC(2048 → 256) + ReLU + Dropout(0.5)
  ├─ FC(256 → 128) + ReLU + Dropout(0.3)
  └─ FC(128 → 8)   ← logits (8 classes)

Total paramètres : ~2.1M
```

**Pourquoi ce choix ?**
- Le spectrogramme mel est une représentation 2D (fréquences × temps) → CNN 2D naturel
- 3 blocs convolutifs pour capturer patterns courts (phonèmes) et longs (intonation)
- BatchNorm + Dropout pour limiter l'overfitting sur ~1440 fichiers
- Global Average Pooling pour robustesse à la durée variable

---

## Résultats attendus

| Métrique | Valeur attendue |
|---|---|
| Val accuracy | ~65–75% |
| Test accuracy | ~60–70% |
| Meilleures classes | happy, angry, neutral |
| Classes difficiles | calm vs neutral |

---

## Points forts pour la soutenance

1. **Dataset réel et original** : RAVDESS avec 24 acteurs professionnels
2. **Feature engineering justifié** : mel spectrogramme vs MFCC (comparaison possible)
3. **Architecture explicable ligne par ligne** : chaque couche a un rôle précis
4. **Démo interactive** : upload audio → prédiction en temps réel + visualisation
5. **Métriques complètes** : accuracy, matrice de confusion, rapport de classification

---

## Questions probables à l'oral

**Q : Pourquoi un CNN et pas un RNN/LSTM ?**
R : Le spectrogramme est une image 2D. Les CNN sont conçus pour capturer des patterns spatiaux locaux, ce qui correspond aux patterns fréquentiels des émotions. Un LSTM traiterait la séquence temporelle brute, ce qui est moins adapté ici.

**Q : Pourquoi le mel spectrogramme ?**
R : L'échelle mel imite la perception humaine des fréquences (logarithmique). Elle compresse les hautes fréquences et étale les basses, ce qui est plus informatif pour la parole que la STFT linéaire.

**Q : Comment gérer l'overfitting avec si peu de données ?**
R : BatchNorm (stabilise l'entraînement), Dropout (régularisation), ReduceLROnPlateau (évite de diverger), data augmentation possible (pitch shift, time stretch via librosa).

**Q : Quelle est la limite principale du modèle ?**
R : La confusion entre calm et neutral (très proches acoustiquement) et la dépendance au locuteur (entraîné sur 24 acteurs en studio, généralisation limitée au monde réel).
