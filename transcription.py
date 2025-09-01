# transcription/google_stt.py
import io
from google.cloud import speech

def transcribe_file(local_path: str, language_code: str = "en-US", sample_rate_hz: int = 8000):
    client = speech.SpeechClient()
    with io.open(local_path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=language_code,
        sample_rate_hertz=sample_rate_hz,
        enable_automatic_punctuation=True,
        model="latest",
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=600)

    text = []
    for result in response.results:
        if result.alternatives:
            text.append(result.alternatives[0].transcript.strip())
    return " ".join(text).strip()
