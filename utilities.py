# utils/audio.py
import os
import numpy as np
import soundfile as sf
import librosa

def load_audio(path: str, target_sr: int = 8000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = np.ascontiguousarray(y, dtype=np.float32)
    return y, target_sr

def peak_normalize(y: np.ndarray, peak: float = 0.99, eps: float = 1e-9):
    m = np.max(np.abs(y)) + eps
    return (y / m) * peak

def save_audio(path: str, y: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")

def chunk_audio(y: np.ndarray, sr: int, chunk_sec: float):
    n = int(chunk_sec * sr)
    if n <= 0:
        return [y]
    chunks = [y[i:i+n] for i in range(0, len(y), n)]
    if len(chunks) and len(chunks[-1]) < n:
        # pad last chunk
        pad = np.zeros(n - len(chunks[-1]), dtype=y.dtype)
        chunks[-1] = np.concatenate([chunks[-1], pad])
    return chunks
