import logging

import torch
from transformers import AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LongTextSummarizationPipeline:

    def __init__(self, model_id: str = "facebook/bart-large-cnn"):
        logging.info("Initializing Text Summarization Pipeline")
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_max_length = self.tokenizer.model_max_length
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            logging.info("Loading model...")
            self.model = pipeline(
                "summarization", model=self.model_id, device=0 if self.device == "cuda" else -1)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30):
        # Check if text needs to be split into smaller chunks
        tokens = self.tokenizer.encode(text, truncation=False, return_tensors='pt')
        if tokens.size(1) > self.model_max_length:
            # Split the text
            return self.summarize_long_text(text, max_length, min_length)
        else:
            return self.model(text, max_length=max_length, min_length=min_length, do_sample=False)

    def summarize_long_text(self, text, max_length, min_length):
        # Split the text into chunks
        chunk_size = self.model_max_length - 50  # To account for [CLS], [SEP], etc.
        text_chunks = self.split_text_into_chunks(text, chunk_size)

        summaries = []
        for chunk in text_chunks:
            summary = self.model(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        return ' '.join(summaries)

    def split_text_into_chunks(self, text, chunk_size):
        tokens = self.tokenizer.encode(text)
        chunk_start = 0
        chunks = []
        while chunk_start < len(tokens):
            chunk_end = min(chunk_start + chunk_size, len(tokens))
            chunk = self.tokenizer.decode(
                tokens[chunk_start:chunk_end], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            chunks.append(chunk)
            chunk_start += chunk_size
        return chunks
