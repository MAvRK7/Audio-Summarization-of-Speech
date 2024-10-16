# Audio Summarization of Speech Signals in Noisy Real-Time Environment using Deep Learning-based Blind Source Separation

This work presents an advanced pipeline for audio source separation,
transcription, and summarization, aimed at enhancing the clarity and intelligibility of speech in multi-speaker environments. The proposed model combines
SepFormer, ConvTasNet, and adaptive noise reduction techniques to isolate
speech from two speakers, reduce background noise, and amplify the primary
speaker’s voice. In contrast to existing models that employ either SepFormer or
ConvTasNet alone, this hybrid approach achieves superior performance without
significantly increasing computational complexity. Once trained, the system delivers rapid, accurate audio separation and transcription. Google’s Speech-to-Text
API is utilised for transcription, followed by a summarization phase implemented
using a pre-trained BART model fine-tuned on the CNN Daily Mail dataset. Performance evaluation using standard metrics, including Signal-to-Distortion Ratio
(SDR), Signal-to-Interference Ratio (SIR), and Signal-to-Artifacts Ratio (SAR),
demonstrates the effectiveness of the proposed model. The system yields high
SDR and SIR scores, indicating its capability in improving speech clarity while
maintaining efficiency. This approach holds promise for various applications such
as meeting transcription, voice command systems, and real-time communication
enhancement.

Keywords: Audio source separation, SepFormer, ConvTasNet, Adaptive noise
reduction, Audio transcription, Summarization
