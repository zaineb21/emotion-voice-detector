# test_mic.py
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

SR = 22050
print("Enregistrement 3 secondes...")
audio = sd.rec(int(3 * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()
print(f"Volume max capté : {np.abs(audio).max():.4f}")

if np.abs(audio).max() < 0.01:
    print("PROBLÈME : micro non détecté ou volume trop bas")
else:
    print("OK : son capté")
    write("test_output.wav", SR, (audio * 32767).astype(np.int16))
    print("Fichier sauvegardé : test_output.wav")