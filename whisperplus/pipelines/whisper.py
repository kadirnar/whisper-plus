import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig, pipeline

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

    def load_model(self, model_id: str = "openai/whisper-large-v3"):
        """
        Loads the pre-trained speech recognition model and moves it to the specified device.

        Args:
            model_id (str): Identifier of the pre-trained model to be loaded.
        """
        logging.info("Loading model...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2",
            device_map="auto")

        logging.info("Model loaded successfully.")

        processor = AutoProcessor.from_pretrained(model_id)

        self.processor = processor
        self.model = model

    def __call__(
            self,
            chunk_length_s: int = 30,
            stride_length_s: int = 5,
            audio_path: str = "test.mp3",
            max_new_tokens: int = 128,
            batch_size: int = 100,
            language: str = "turkish"):
        """
        Converts audio to text using the pre-trained speech recognition model.

        Args:
            audio_path (str): Path to the audio file to be transcribed.

        Returns:
            str: Transcribed text from the audio.
        """
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            max_new_tokens=max_new_tokens,
            batch_size=100,
            device_map="auto",
            return_timestamps=True,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            model_kwargs={"use_flash_attention_2": True},
            generate_kwargs={"language": language},
        )
        logging.info("Transcribing audio...")
        result = pipe(audio_path)
        return result
