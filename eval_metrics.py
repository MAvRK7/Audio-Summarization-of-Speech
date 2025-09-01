# evaluation/metrics.py
import numpy as np
from typing import Dict, Optional

def si_snr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    proj = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    noise = est - proj
    ratio = (np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return 10 * np.log10(ratio + eps)

def bss_eval(est_sources: np.ndarray, ref_sources: np.ndarray) -> Dict[str, float]:
    """
    est_sources, ref_sources: [n_src, time]
    Returns average SDR/SIR/SAR across sources.
    """
    try:
        import mir_eval
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref_sources, est_sources, compute_permutation=True)
        return {
            "SDR": float(np.mean(sdr)),
            "SIR": float(np.mean(sir)),
            "SAR": float(np.mean(sar)),
        }
    except Exception:
        return {"SDR": float("nan"), "SIR": float("nan"), "SAR": float("nan")}

def evaluate_pair(est1: np.ndarray, est2: np.ndarray, ref1: Optional[np.ndarray], ref2: Optional[np.ndarray]) -> Dict[str, float]:
    metrics = {}
    if ref1 is not None and ref2 is not None and len(ref1) == len(est1) and len(ref2) == len(est2):
        bss = bss_eval(np.stack([est1, est2], axis=0), np.stack([ref1, ref2], axis=0))
        metrics.update(bss)
        metrics["SI-SNR"] = float(np.mean([si_snr(est1, ref1), si_snr(est2, ref2)]))
    else:
        metrics["SDR"] = float("nan")
        metrics["SIR"] = float("nan")
        metrics["SAR"] = float("nan")
        metrics["SI-SNR"] = float("nan")
    return metrics
