# noise_reduction/denoise.py
import numpy as np
import noisereduce as nr

def adaptive_denoise(y: np.ndarray, sr: int, prop_decrease: float = 0.9, stationary: bool = False, n_fft: int = 1024, hop_length: int = 256):
    return nr.reduce_noise(
        y=y,
        sr=sr,
        prop_decrease=prop_decrease,
        stationary=stationary,
        n_fft=n_fft,
        hop_length=hop_length,
        use_tqdm=False,
    ).astype(np.float32)
