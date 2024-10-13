# Audio Summarization of Speech Signals in Noisy Real-Time Environment using Deep Learning-based Blind Source Separation

Speech signal processing in noisy, real-time environments poses significant challenges due to overlapping audio sources and diverse noise types. This work presents a
deep learning-based solution for summarizing speech signals
by combining three powerful techniques: ConvTasnet for time-
domain speech separation, Sepformer for transformer-based
separation of overlapping speakers, and Adaptive Noise Filter
ing for dynamic noise reduction. The proposed methodology
effectively separates speech from noise and competing speak
ers, producing clear and concise audio summaries. Evaluations
on standard datasets demonstrate superior performance over
traditional methods, with significant improvements in Signal-
to-Distortion Ratio (SDR) and Perceptual Evaluation of Speech
Quality (PESQ). This framework is ideal for real-time applica
tions like teleconferencing and assistive listening devices, offering
an efficient and scalable solution to the challenges of noisy speech
processing. The increasing demand for robust speech processing
systems in noisy real-time environments has highlighted the need
for more advanced solutions capable of handling overlapping
speech and fluctuating noise levels. This work introduces a
hybrid deep learning framework that leverages ConvTasnet’s
capability to process time-domain audio data and Sepformer’s
powerful transformer-based architecture to achieve effective
speech source separation. In addition to these, adaptive noise
filtering further refines the audio output, minimizing noise
artifacts and enhancing clarity. The performance of the model is
validated through extensive experimentation using real-world
noisy datasets, outperforming existing methods in terms of both
quantitative and qualitative assessments. These results underline
the potential application of the proposed approach in various
real-time systems such as hearing aids, real-time communication
platforms, and voice-controlled AI systems where noise and
speech overlap are major challenges.
