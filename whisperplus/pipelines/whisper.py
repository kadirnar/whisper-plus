import logging
from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from whisperplus.model.load_model import load_model_whisper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpeechToTextPipeline:
    """Class for converting audio to text using a pre-trained speech recognition model."""

    def __init__(
        self,
        model_id: str = "distil-whisper/distil-large-v3",
        quant_config=None,
        flash_attention_2: Optional[bool] = True,
        hqq_compile: Optional[bool] = False,
    ):
        self.model = None
        self.device = "cuda"
        self.hqq_compile = hqq_compile
        self.flash_attention_2 = flash_attention_2

        if self.model is None:
            self.load_plus_model(model_id, quant_config, hqq_compile, flash_attention_2)
        else:
            logging.info("Model already loaded.")

    def load_plus_model(
        self,
        model_id: str = "distil-whisper/distil-large-v3",
        quant_config=None,
        hqq_compile: bool = False,
        flash_attention_2: bool = True,
    ):

        model, processor = load_model_whisper(
            model_id=model_id,
            quant_config=quant_config,
            hqq_compile=hqq_compile,
            flash_attention_2=flash_attention_2,
            device=self.device)

        self.model = model
        self.processor = processor

        return model

    def __call__(
            self,
            audio_path: str = "test.mp3",
            chunk_length_s: int = 30,
            stride_length_s: int = 5,
            max_new_tokens: int = 128,
            batch_size: int = 100,
            language: str = "turkish",
            return_timestamps: bool = False):
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
            batch_size=batch_size,
            return_timestamps=return_timestamps,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            model_kwargs={"use_flash_attention_2": self.flash_attention_2},
            generate_kwargs={"language": language},
            device_map=self.device)
        logging.info("Transcribing audio...")
        result = pipe(audio_path)
        return result
