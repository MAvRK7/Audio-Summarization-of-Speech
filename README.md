# Multi-Speaker Speech Processing in Noisy Environments: A Hybrid Model for Source Separation and Summarization

In multi-speaker environments, intelligibility may be a concern when
speakers overlap. This work presents an advanced pipeline to first separate audio
and then give a summary of the conversation. The proposed model combines Sep
Former, ConvTasNet, and adaptive noise reduction techniques to isolate speech
from two speaker mixed audio, reduce background noise, and amplify the pri
mary speaker’s voice. This hybrid approach gives better results than each of the
two models used on their own, without significant increase in computational cost.
Once trained, the system delivers rapid, accurate audio separation and transcrip
tion. Once separated, for transcipting the audio, Google’s Speech-to-Text API
is utilised. This is followed by a summarization phase implemented using a pre-
trained BART model fine-tuned on the CNN Daily Mail dataset. Performance
evaluation is done using standard metrics, including Signal-to-Distortion Ratio
(SDR), Signal-to-Interference Ratio (SIR), and Signal-to-Artifacts Ratio (SAR)
and it demonstrates the effectiveness of the proposed model. The model yields
an average SDR of 24.6, average SIR of 24.5 and an average SAR of 24.5 which
shows its capability in improving speech clarity while maintaining efficiency.

Keywords: Audio source separation, SepFormer, ConvTasNet, Adaptive noise
reduction, Audio transcription, Summarization
