# Author: @GuldenizBektas # ignore [E265]

import logging

import torch
from scipy.io.wavfile import write as write_wav
from transformers import AutoModelForTextToWaveform, AutoProcessor


class TextToSpeechPipeline:
    """Class to convert text to speech."""

    def __init__(self, model_id: str = "suno/bark"):
        self.model_id = model_id
        self.model = None
        self.device = None

        if self.model is None:
            self.load_model(model_id)
        else:
            logging.info("Model is already loaded.")

        self.set_device()

    def set_device(self):
        """Set device for model."""
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

    def load_model(self, model_id: str = "suno/bark"):
        logging.info("Loading Model...")

        model = AutoModelForTextToWaveform.from_pretrained(model_id)

        model.to(self.device)  # type: ignore
        logging.info("Model loaded 🎉")
        self.model = model

    def __call__(self, text: str, voice_preset: str = "v2/en_speaker_6"):
        processor = AutoProcessor.from_pretrained(
            self.model_id, torch_dtype=torch.float16, use_flash_attention_2=True, device=self.device)
        inputs = processor(text, voice_preset)
        outputs = self.model.generate(**inputs)  # type: ignore

        audio_array = outputs.cpu().numpy().squeeze()

        sample_rate = self.model.generation_config.sample_rate
        write_wav("bark_generation.wav", sample_rate, audio_array)
        logging.info("bark_generation.wav saved 🎉")

        return audio_array
