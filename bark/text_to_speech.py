import logging

from IPython.display import Audio

from bark import SAMPLE_RATE, generate_audio, preload_models


class TextToSpeechPipeline:

    def __init__(self, input: str):
        self.input = None

        if self.input is None:
            logging.warning("No text found")
            return None

    def convert_to_speech(self, text):
        self.input = text
        audio_array = generate_audio(self.input)
        Audio(audio_array, rate=SAMPLE_RATE)
