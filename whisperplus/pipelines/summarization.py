import logging

import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TextSummarizationPipeline:

    def __init__(self, model_id: str = "facebook/bart-large-cnn"):
        logging.info("Initializing Text Summarization Pipeline")
        self.model_id = model_id
        self.model = None
        self.set_device()

    def set_device(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logging.info(f"Using device: {self.device}")

    def load_model(self):
        if self.model is None:
            try:
                logging.info("Loading model...")
                self.model = pipeline("summarization", model=self.model_id, device=self.device)
                logging.info("Model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading model: {e}")

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30):
        if self.model is None:
            self.load_model()

        logging.info("Performing text summarization")
        return self.model(text, max_length=max_length, min_length=min_length, do_sample=False)
