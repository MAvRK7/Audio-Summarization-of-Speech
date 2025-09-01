# summarization/bart_summarizer.py
from transformers import pipeline

class BartSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.pipe = pipeline("summarization", model=model_name)

    def _chunk(self, text: str, max_chars: int = 6000, overlap: int = 200):
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunks.append(text[start:end])
            if end == len(text): break
            start = end - overlap
        return chunks

    def summarize(self, text: str, max_chars: int = 6000, overlap: int = 200, gen_kwargs: dict | None = None):
        gen_kwargs = gen_kwargs or {}
        chunks = self._chunk(text, max_chars, overlap)
        partials = [self.pipe(c, **gen_kwargs)[0]["summary_text"] for c in chunks]
        merged = " ".join(partials)
        # Optional: one more pass to tighten
        final = self.pipe(merged, **gen_kwargs)[0]["summary_text"]
        return final
