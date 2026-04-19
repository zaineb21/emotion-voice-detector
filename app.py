"""
app.py — Démo Streamlit : détection d'émotion dans la voix
Lancer : streamlit run app.py
"""

import os
import time
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import streamlit as st

from model import EmotionCNN
from dataset import EMOTIONS, N_CLASSES, extract_mel_spectrogram


# ─── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Voice Detector",
    page_icon="🎙️",
    layout="centered",
)

CHECKPOINT_PATH = "./checkpoints/best_model.pth"
EMOTION_EMOJIS = {
    "neutral":   "😐",
    "calm":      "😌",
    "happy":     "😄",
    "sad":       "😢",
    "angry":     "😠",
    "fearful":   "😨",
    "disgust":   "🤢",
    "surprised": "😲",
}
EMOTION_COLORS = {
    "neutral":   "#8888AA",
    "calm":      "#66BB6A",
    "happy":     "#FDD835",
    "sad":       "#42A5F5",
    "angry":     "#EF5350",
    "fearful":   "#AB47BC",
    "disgust":   "#26A69A",
    "surprised": "#FF7043",
}
TMP_RECORD = os.path.join(os.path.expanduser("~"), "recorded.wav")
SR = 22050


# ─── Modèle ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = EmotionCNN(n_classes=N_CLASSES)
    if not os.path.exists(CHECKPOINT_PATH):
        st.error("Modèle introuvable. Lance d'abord : python train.py")
        st.stop()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


def predict(file_path, model, device):
    feat = extract_mel_spectrogram(file_path)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model.predict_proba(x).squeeze().cpu().numpy()
    return EMOTIONS[np.argmax(probs)], probs


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_mel(file_path):
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                   y_axis="mel", ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogramme")
    fig.tight_layout()
    return fig


def plot_proba(probs):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(EMOTIONS, probs * 100,
                   color=[EMOTION_COLORS[e] for e in EMOTIONS], edgecolor="none")
    ax.set_xlabel("Probabilité (%)"); ax.set_xlim(0, 100)
    ax.set_title("Distribution des émotions")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


# ─── Résultats ────────────────────────────────────────────────────────────────
def show_results(tmp_path, model, device):
    with open(tmp_path, "rb") as f:
        st.audio(f.read(), format="audio/wav")
    with st.spinner("Analyse en cours..."):
        emotion, probs = predict(tmp_path, model, device)

    color = EMOTION_COLORS.get(emotion, "#888")
    emoji = EMOTION_EMOJIS.get(emotion, "")
    st.markdown(f"""
        <div style="background:{color}22;border-left:5px solid {color};
                    border-radius:8px;padding:16px 20px;margin:16px 0;">
            <h2 style="margin:0;color:{color}">{emoji} Émotion : <strong>{emotion.upper()}</strong></h2>
            <p style="margin:4px 0 0;color:#888">Confiance : <strong>{max(probs)*100:.1f}%</strong></p>
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spectrogramme mel")
        st.pyplot(plot_mel(tmp_path))
    with col2:
        st.subheader("Probabilités")
        st.pyplot(plot_proba(probs))

    st.subheader("Top 3")
    for rank, idx in enumerate(np.argsort(probs)[::-1][:3], 1):
        emo = EMOTIONS[idx]
        st.progress(float(probs[idx]),
                    text=f"{rank}. {EMOTION_EMOJIS.get(emo,'')} {emo} — {probs[idx]*100:.1f}%")


# ─── Interface ────────────────────────────────────────────────────────────────
def main():
    st.title("🎙️ Emotion Voice Detector")
    st.markdown("Enregistre ta voix ou uploade un fichier — le modèle détecte l'**émotion** instantanément.")

    model, device = load_model()

    tab1, tab2, tab3 = st.tabs(["🎤 Enregistrer", "📂 Uploader", "🎭 Exemples RAVDESS"])

    # ── Tab 1 : Enregistrement micro ──────────────────────────────────────────
    with tab1:
        st.markdown("Choisis une durée et appuie sur **Enregistrer**.")

        duration = st.slider("Durée d'enregistrement (secondes)", 2, 6, 3, key="dur")

        if st.button("🎤 Enregistrer", key="record_btn"):
            placeholder = st.empty()

            # Compte à rebours
            for i in range(duration, 0, -1):
                placeholder.markdown(f"### 🔴 Enregistrement... {i}s")
                time.sleep(1)
            placeholder.markdown("### ⏳ Traitement...")

            # Enregistrement
            audio_data = sd.rec(
                int(duration * SR),
                samplerate=SR,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            # Amplification + normalisation
            audio_flat = audio_data.flatten()
            max_val = np.abs(audio_flat).max()
            if max_val > 0:
                audio_flat = audio_flat / max_val * 0.9
            import soundfile as sf
            sf.write(TMP_RECORD, audio_flat, SR, subtype="PCM_16")
            placeholder.empty()
            st.success("Enregistrement terminé !")
            show_results(TMP_RECORD, model, device)

    # ── Tab 2 : Upload ─────────────────────────────────────────────────────────
    with tab2:
        uploaded = st.file_uploader("Choisir un fichier audio", type=["wav", "mp3", "ogg"])
        if uploaded is not None:
            tmp_path = os.path.join(os.path.expanduser("~"), uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            show_results(tmp_path, model, device)

    # ── Tab 3 : Exemples RAVDESS ──────────────────────────────────────────────
    with tab3:
        example_dir = "./data/RAVDESS/Actor_01"
        if os.path.isdir(example_dir):
            examples = [f for f in os.listdir(example_dir) if f.endswith(".wav")][:6]
            cols = st.columns(3)
            selected_ex = None
            for i, ex in enumerate(examples):
                parts = ex.replace(".wav", "").split("-")
                emo_name = {
                    "01": "neutral", "02": "calm",     "03": "happy",
                    "04": "sad",     "05": "angry",    "06": "fearful",
                    "07": "disgust", "08": "surprised"
                }.get(parts[2], "?")
                if cols[i % 3].button(
                    f"{EMOTION_EMOJIS.get(emo_name,'?')} {emo_name}", key=f"ex_{i}"
                ):
                    selected_ex = os.path.join(example_dir, ex)
            if selected_ex:
                show_results(selected_ex, model, device)
        else:
            st.info("Exemples non disponibles (Actor_01 introuvable).")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## À propos")
        st.markdown(
            "**Modèle** : CNN 2D (PyTorch)\n\n"
            "**Dataset** : RAVDESS — 1440 fichiers\n\n"
            "**Features** : Mel spectrogramme 128×128\n\n"
            "**Classes** : 8 émotions"
        )
        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
            st.success(
                f"Modèle chargé\n\n"
                f"Epoch : {ckpt['epoch']} | Val acc : {ckpt['val_acc']*100:.1f}%"
            )


if __name__ == "__main__":
    main()