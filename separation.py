# separation/sepformer_infer.py
import torch
import numpy as np

class SepformerSeparator:
    def __init__(self, source: str, savedir: str, device: str = "cuda", backend: str = "speechbrain"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.model = None
        if backend == "speechbrain":
            from speechbrain.pretrained import SepformerSeparation as SB_Sep
            self.model = SB_Sep.from_hparams(source=source, savedir=savedir, run_opts={"device": str(self.device)})
        elif backend == "local":
            raise NotImplementedError("Local SepFormer weights/arch not provided. Use backend='speechbrain'.")
        else:
            raise ValueError("Unsupported backend for SepFormer.")

    @torch.inference_mode()
    def separate(self, y: np.ndarray, sr: int):
        # SpeechBrain expects torch tensor [1, time]
        import torch
        wav = torch.from_numpy(y).float().unsqueeze(0).to(self.device)
        est_sources = self.model.separate_batch(wav)  # shape: [batch, n_src, time]
        est_sources = est_sources.squeeze(0).cpu().numpy()  # [n_src, time]
        # Return two sources; if model outputs >2 sources, take first two
        if est_sources.shape[0] < 2:
            # duplicate if only single source head (unlikely)
            est_sources = np.vstack([est_sources, np.zeros_like(est_sources)])
        return est_sources[0], est_sources[1]
# separation/convtasnet_infer.py
import torch
import numpy as np

class ConvTasNetRefiner:
    """
    Acts as a refinement/denoising stage by running ConvTasNet on each separated source
    and selecting the most correlated output per source.
    """
    def __init__(self, source: str, device: str = "cuda", backend: str = "asteroid"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.model = None
        if backend == "asteroid":
            from asteroid.models import ConvTasNet
            self.model = ConvTasNet.from_pretrained(source).to(self.device).eval()
        elif backend == "local":
            raise NotImplementedError("Provide local ConvTasNet checkpoint and architecture.")
        elif backend == "disabled":
            pass
        else:
            raise ValueError("Unsupported backend for ConvTasNet.")

    @torch.inference_mode()
    def refine_pair(self, s1: np.ndarray, s2: np.ndarray, sr: int):
        if self.model is None:
            return s1, s2  # refinement disabled

        import torch
        # Stack and recombine to let ConvTasNet operate on pair
        # Build a pseudo-mixture from the two separated sources (residual artifacts will be redistributed)
        mix = s1 + s2
        x = torch.from_numpy(mix).float().unsqueeze(0).to(self.device)  # [1, time]
        estimates = self.model(x)  # [1, n_src, time]
        estimates = estimates.squeeze(0).cpu().numpy()

        # If n_src >= 2, align outputs to original sources via correlation
        if estimates.shape[0] >= 2:
            e1, e2 = estimates[0], estimates[1]
            # Assign by maximizing correlation with original s1/s2
            def corr(a, b):
                a = a - a.mean()
                b = b - b.mean()
                denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                return float(np.dot(a, b) / denom)

            combos = [
                (corr(e1, s1) + corr(e2, s2), (e1, e2)),
                (corr(e2, s1) + corr(e1, s2), (e2, e1)),
            ]
            _, (r1, r2) = max(combos, key=lambda z: z[0])
            return r1, r2
        else:
            # Single-source head: treat as denoiser; return denoised s1, leave s2 unchanged
            return estimates[0], s2
