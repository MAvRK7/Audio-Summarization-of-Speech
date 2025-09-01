# run_pipeline.py
import os
import argparse
import yaml
from tqdm import tqdm

from utils.audio import load_audio, peak_normalize, save_audio
from separation.sepformer_infer import SepformerSeparator
from separation.convtasnet_infer import ConvTasNetRefiner
from noise_reduction.denoise import adaptive_denoise
from transcription.google_stt import transcribe_file
from summarization.bart_summarizer import BartSummarizer
from evaluation.metrics import evaluate_pair

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input_audio", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load + preprocess
    y, sr = load_audio(args.input_audio, target_sr=cfg["audio"]["target_sr"])
    y = peak_normalize(y, cfg["audio"]["normalize_peak"])

    # SepFormer
    sep = SepformerSeparator(
        source=cfg["models"]["sepformer"]["source"],
        savedir=cfg["models"]["sepformer"]["savedir"],
        device=cfg["models"]["sepformer"]["device"],
        backend=cfg["models"]["sepformer"]["backend"],
    )
    s1, s2 = sep.separate(y, sr)

    # ConvTasNet refinement
    ctn = ConvTasNetRefiner(
        source=cfg["models"]["convtasnet"]["source"],
        device=cfg["models"]["convtasnet"]["device"],
        backend=cfg["models"]["convtasnet"]["backend"],
    )
    s1_ref, s2_ref = ctn.refine_pair(s1, s2, sr)

    # Adaptive noise reduction
    if cfg["noise_reduction"]["enabled"]:
        s1_ref = adaptive_denoise(
            s1_ref, sr,
            prop_decrease=cfg["noise_reduction"]["prop_decrease"],
            stationary=cfg["noise_reduction"]["stationary"],
            n_fft=cfg["noise_reduction"]["n_fft"],
            hop_length=cfg["noise_reduction"]["hop_length"],
        )
        s2_ref = adaptive_denoise(
            s2_ref, sr,
            prop_decrease=cfg["noise_reduction"]["prop_decrease"],
            stationary=cfg["noise_reduction"]["stationary"],
            n_fft=cfg["noise_reduction"]["n_fft"],
            hop_length=cfg["noise_reduction"]["hop_length"],
        )

    # Save separated outputs
    os.makedirs(args.out_dir, exist_ok=True)
    out_s1 = os.path.join(args.out_dir, "separated_audio_1.wav")
    out_s2 = os.path.join(args.out_dir, "separated_audio_2.wav")
    save_audio(out_s1, s1_ref, sr)
    save_audio(out_s2, s2_ref, sr)

    # Transcription
    transcript = ""
    if cfg["stt"]["enabled"]:
        # Google STT prefers 16-bit PCM; utils.save_audio already writes LINEAR16 via soundfile PCM_16
        t1 = transcribe_file(out_s1, language_code=cfg["stt"]["language_code"], sample_rate_hz=cfg["stt"]["sample_rate_hz"])
        t2 = transcribe_file(out_s2, language_code=cfg["stt"]["language_code"], sample_rate_hz=cfg["stt"]["sample_rate_hz"])
        transcript = f"Speaker 1: {t1}\nSpeaker 2: {t2}"
        with open(os.path.join(args.out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)

    # Summarization
    if cfg["summarization"]["enabled"] and transcript.strip():
        summarizer = BartSummarizer(cfg["summarization"]["model_name"])
        summary = summarizer.summarize(
            transcript,
            max_chars=cfg["summarization"]["max_input_chars"],
            overlap=cfg["summarization"]["chunk_overlap"],
            gen_kwargs=cfg["summarization"]["gen_kwargs"],
        )
        with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary)

    # Evaluation (optional, needs references)
    if cfg["evaluation"]["enabled"]:
        ref1 = cfg["evaluation"].get("ref_s1") or ""
        ref2 = cfg["evaluation"].get("ref_s2") or ""
        r1 = r2 = None
        if os.path.isfile(ref1) and os.path.isfile(ref2):
            import librosa, numpy as np
            import soundfile as sf
            from utils.audio import load_audio
            r1, _ = load_audio(ref1, cfg["audio"]["target_sr"])
            r2, _ = load_audio(ref2, cfg["audio"]["target_sr"])
            # align lengths
            minlen = min(len(r1), len(s1_ref), len(r2), len(s2_ref))
            r1, r2 = r1[:minlen], r2[:minlen]
            s1_m, s2_m = s1_ref[:minlen], s2_ref[:minlen]
            metrics = evaluate_pair(s1_m, s2_m, r1, r2)
        else:
            metrics = {"SDR": float("nan"), "SIR": float("nan"), "SAR": float("nan"), "SI-SNR": float("nan")}

        print("Metrics:", metrics)
        if cfg["evaluation"]["write_csv"]:
            import csv
            csv_path = cfg["evaluation"]["output_csv"]
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            write_header = not os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["input", "SDR", "SIR", "SAR", "SI-SNR"])
                if write_header: w.writeheader()
                row = {"input": args.input_audio} | metrics
                w.writerow(row)

    print(f"Done. Outputs written to: {args.out_dir}")

if __name__ == "__main__":
    main()
