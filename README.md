# 🎙️ Multi‑Speaker Speech Processing in Noisy Environments

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)]()
[![Conference](https://img.shields.io/badge/Accepted%20at-SPIN%202025-blueviolet)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📌 Overview
This is a hybrid speech processing pipeline designed to:
1. **Separate** overlapping speech from noisy, multi‑speaker audio
2. **Transcribe** each speaker’s voice
3. **Summarize** the conversation

The system integrates:
- **SepFormer** — transformer‑based speech separation
- **ConvTasNet** — convolutional time‑domain separation
- **Adaptive noise reduction** — to suppress background noise and enhance intelligibility
- **Google Speech‑to‑Text API** — for transcription
- **BART (fine‑tuned)** — for abstractive summarization

---

## 🛰 Conference & Publication
- **Accepted at**: *International Conference on Signal Processing and Integrated Networks (SPIN 2025)*
- **To be published in**: *Lecture Notes in Electrical Engineering* (Springer)

---

## 🎧 Example Audio

| File | Description |
|------|-------------|
| `mixed_audio.mp3` | Original noisy, overlapping two‑speaker audio |
| `separated_audio_1.wav` | Speaker 1 isolated |
| `separated_audio_2.wav` | Speaker 2 isolated |

---

## 🧪 Methodology

**Pipeline Flow**:

<img width="203" height="506" alt="image" src="https://github.com/user-attachments/assets/dfdb6211-f177-42ac-9563-9411ed4781c1" />


---

## 📊 Performance

| Metric | Score (avg) |
|--------|-------------|
| SDR (Signal‑to‑Distortion Ratio) | 24.6 |
| SIR (Signal‑to‑Interference Ratio) | 24.5 |
| SAR (Signal‑to‑Artifacts Ratio) | 24.5 |

---

## 🚀 Key Features
- **Hybrid separation**: Combines transformer and convolutional models
- **Noise‑aware**: Adaptive filtering for real‑world conditions
- **End‑to‑end**: From raw noisy audio to summarized conversation
- **Lightweight**: No significant increase in computational cost

---

## 💡 Skills Demonstrated
Speech separation (SepFormer, ConvTasNet)

Noise reduction algorithms

Speech‑to‑text integration

Abstractive summarization (BART fine‑tuning)

Audio signal processing & evaluation metrics

End‑to‑end ML pipeline design

---

## 🙏 Acknowledgments
SepFormer & ConvTasNet authors

Google Speech‑to‑Text API

BART model (Hugging Face Transformers)

SPIN 2025 Conference Committee
