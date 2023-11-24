import logging

import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import write

from bark import SAMPLE_RATE, generate_audio, preload_models


class TextToSpeechPipeline:

    def __init__(self, input_text: str, output_path: str = "output.mp3"):
        self.input_text = input_text
        self.output_path = output_path

        if not self.input_text:
            logging.warning("Input text is empty. Cannot convert to speech.")
            # You may raise an exception or return a specific value if needed

    def convert_to_speech(self):
        if not self.input_text:
            logging.warning("No text to convert to speech. Aborting.")
            return

        audio_array = generate_audio(self.input_text)
        # Convert NumPy array to AudioSegment
        # Convert NumPy array to AudioSegment
        audio_array = (audio_array * 32767).astype(np.int16)  # type: ignore # Scale to 16-bit integer
        audio_segment = AudioSegment(
            audio_array.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

        audio_segment.export(self.output_path, format="mp3")

        print(f"Speech saved to {self.output_path}")
