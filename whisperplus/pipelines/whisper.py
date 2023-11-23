import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpeechToTextPipeline:
    """Class for converting audio to text using a pre-trained speech recognition model."""

    def __init__(self, model_id: str = "openai/whisper-large-v3"):
        self.model = None
        self.device = None

        if self.model is None:
            self.load_model(model_id)
        else:
            logging.info("Model already loaded.")

        self.set_device()

    def set_device(self):
        """Sets the device to be used for inference based on availability."""
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

    def load_model(self, model_id: str = "openai/whisper-large-v3"):
        """
        Loads the pre-trained speech recognition model and moves it to the specified device.

        Args:
            model_id (str): Identifier of the pre-trained model to be loaded.
        """
        logging.info("Loading model...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
        model.to(self.device)
        logging.info("Model loaded successfully.")

        self.model = model

    def __call__(self, audio_path: str, model_id: str = "openai/whisper-large-v3", language: str = "turkish"):
        """
        Converts audio to text using the pre-trained speech recognition model.

        Args:
            audio_path (str): Path to the audio file to be transcribed.
            model_id (str): Identifier of the pre-trained model to be used for transcription.

        Returns:
            str: Transcribed text from the audio.
        """
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            torch_dtype=torch.float16,
            chunk_length_s=30,
            max_new_tokens=128,
            batch_size=24,
            return_timestamps=True,
            device=self.device,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            model_kwargs={"use_flash_attention_2": True},
            generate_kwargs={"language": language},
        )
        logging.info("Transcribing audio...")
        result = pipe(audio_path)["text"]
        return result
